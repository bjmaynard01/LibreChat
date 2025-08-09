const { sendEvent } = require('@librechat/api');
const { logger } = require('@librechat/data-schemas');
const { Constants } = require('librechat-data-provider');
const {
  handleAbortError,
  createAbortController,
  cleanupAbortController,
} = require('~/server/middleware');
const { disposeClient, clientRegistry, requestDataMap } = require('~/server/cleanup');
const { saveMessage } = require('~/models');

const { getRAGContext } = require('~/server/controllers/assistants/chatV2');

const AgentController = async (req, res, next, initializeClient, addTitle) => {
  let {
    text,
    isRegenerate,
    endpointOption,
    conversationId,
    isContinued = false,
    editedContent = null,
    parentMessageId = null,
    overrideParentMessageId = null,
    responseMessageId: editedResponseMessageId = null,
  } = req.body;

  const userId = req.user.id;
  const newConvo = !conversationId;

  let sender,
    abortKey,
    userMessage,
    userMessageId,
    userMessagePromise,
    promptTokens,
    responseMessageId,
    client = null,
    getReqData,
    getAbortData,
    cleanupHandlers = [];

  getReqData = (data = {}) => {
    if ('userMessage' in data) {
      userMessage = data.userMessage;
      userMessageId = data.userMessage.messageId;
    }
    if ('userMessagePromise' in data) userMessagePromise = data.userMessagePromise;
    if ('responseMessageId' in data) responseMessageId = data.responseMessageId;
    if ('promptTokens' in data) promptTokens = data.promptTokens;
    if ('sender' in data) sender = data.sender;
    if (!conversationId && 'conversationId' in data) conversationId = data.conversationId;
  };

  const performCleanup = () => {
    logger.debug('[AgentController] Performing cleanup');
    if (Array.isArray(cleanupHandlers)) {
      for (const handler of cleanupHandlers) {
        try {
          if (typeof handler === 'function') handler();
        } catch (e) {
          logger.error('[AgentController] Cleanup handler error', e);
        }
      }
    }
    if (abortKey) cleanupAbortController(abortKey);
    if (client) disposeClient(client);

    client = null;
    getReqData = null;
    userMessage = null;
    getAbortData = null;
    endpointOption.agent = null;
    endpointOption = null;
    cleanupHandlers = null;
    userMessagePromise = null;

    if (requestDataMap.has(req)) requestDataMap.delete(req);
    logger.debug('[AgentController] Cleanup complete');
  };

  try {
    const result = await initializeClient({ req, res, endpointOption });
    client = result.client;

    if (clientRegistry) clientRegistry.register(client, { userId }, client);
    requestDataMap.set(req, { client });

    const contentRef = new WeakRef(client.contentParts || []);
    getAbortData = () => ({
      sender,
      content: contentRef.deref() || [],
      userMessage,
      promptTokens,
      conversationId,
      userMessagePromise,
      messageId: responseMessageId,
      parentMessageId: overrideParentMessageId ?? userMessageId,
    });

    const { abortController, onStart } = createAbortController(req, res, getAbortData, getReqData);

    const closeHandler = () => {
      if (!abortController?.signal?.aborted && !abortController?.requestCompleted) {
        abortController.abort();
        logger.debug('[AgentController] Request aborted on close');
      }
    };

    res.on('close', closeHandler);
    cleanupHandlers.push(() => res.removeListener('close', closeHandler));

    const messageOptions = {
      user: userId,
      onStart,
      getReqData,
      isContinued,
      isRegenerate,
      editedContent,
      conversationId,
      parentMessageId,
      abortController,
      overrideParentMessageId,
      isEdited: !!editedContent,
      responseMessageId: editedResponseMessageId,
      progressOptions: { res },
    };

    // 👇 Inject context ONLY into model's prompt
    let contextText = '';
    try {
      contextText = await getRAGContext(text);
      if (contextText) logger.info(`🧩 Injected RAG context:\n${contextText.slice(0, 1000)}...`);
      else logger.info('📭 No RAG context injected');
    } catch (e) {
      logger.error('❌ Error getting RAG context:', e);
    }

    /*const modifiedText = contextText
      ? `Use the following context to answer the user’s question. If the context is insufficient, say so.\n\n${contextText}\n\nQuestion: ${text}`
      : text;

    const response = await client.sendMessage(modifiedText, messageOptions);*/
    // NEW WAY — inject context as system instruction
    const promptPrefix = contextText
      ? `Use the following context to answer the user’s question. If the context is insufficient, say so.\n\n${contextText}`
      : undefined;
      
    const response = await client.sendMessage(text, {
      ...messageOptions,
      promptPrefix,
    });

    const messageId = response.messageId;
    const endpoint = endpointOption.endpoint;
    response.endpoint = endpoint;

    const databasePromise = response.databasePromise;
    delete response.databasePromise;

    const { conversation: convoData = {} } = await databasePromise;
    const conversation = { ...convoData };
    conversation.title =
      conversation && !conversation.title ? null : conversation?.title || 'New Chat';

    // --- Save user-visible message with clean user text
    const visibleUserMessage = {
      role: 'user',
      content: [{ type: 'text', text }],
      user: userId,
      messageId: userMessageId,
      parentMessageId,
      conversationId,
      metadata: {
        contextInjected: !!contextText,
      },
    };
    userMessage = visibleUserMessage;

    // Attach files if needed
    if (req.body.files && client.options?.attachments) {
      visibleUserMessage.files = [];
      const fileIds = new Set(req.body.files.map((file) => file.file_id));
      for (const att of client.options.attachments) {
        if (fileIds.has(att.file_id)) visibleUserMessage.files.push({ ...att });
      }
    }

    // Send streaming response
    if (!abortController.signal.aborted) {
      sendEvent(res, {
        final: true,
        conversation,
        title: conversation.title,
        requestMessage: visibleUserMessage,
        responseMessage: { ...response },
      });
      res.end();

      if (client.savedMessageIds && !client.savedMessageIds.has(messageId)) {
        await saveMessage(
          req,
          { ...response, user: userId },
          { context: 'AgentController: response end' },
        );
      }
    }

    if (!client.skipSaveUserMessage) {
      await saveMessage(req, visibleUserMessage, {
        context: 'AgentController: saving user message',
      });
    }

    if (addTitle && parentMessageId === Constants.NO_PARENT && newConvo) {
      addTitle(req, { text, response: { ...response }, client })
        .catch((err) => logger.error('AgentController: title error', err))
        .finally(performCleanup);
    } else {
      performCleanup();
    }
  } catch (err) {
    handleAbortError(res, req, err, {
      conversationId,
      sender,
      messageId: responseMessageId,
      parentMessageId: overrideParentMessageId ?? userMessageId ?? parentMessageId,
      userMessageId,
    })
      .catch((err2) => logger.error('AgentController: abort error', err2))
      .finally(performCleanup);
  }
};

module.exports = AgentController;
