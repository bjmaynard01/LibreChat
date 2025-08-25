const { sendEvent } = require('@librechat/api');
const { logger } = require('@librechat/data-schemas');
const { Constants } = require('librechat-data-provider');
const {
  handleAbortError,
  createAbortController,
  cleanupAbortController,
} = require('~/server/middleware');
const { disposeClient, clientRegistry, requestDataMap } = require('~/server/cleanup');
const { getRagContext, formatRagContext } = require('~/server/services/rag');
const { saveMessage } = require('~/models');

const DEBUG = process.env.RAG_DEBUG === 'true';

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

  let sender;
  let abortKey;
  let userMessage;
  let promptTokens;
  let userMessageId;
  let responseMessageId;
  let userMessagePromise;
  let getAbortData;
  let client = null;
  let cleanupHandlers = [];

  const newConvo = !conversationId;
  const userId = req.user.id;

  let getReqData = (data = {}) => {
    for (let key in data) {
      if (key === 'userMessage') {
        userMessage = data[key];
        userMessageId = data[key].messageId;
      } else if (key === 'userMessagePromise') {
        userMessagePromise = data[key];
      } else if (key === 'responseMessageId') {
        responseMessageId = data[key];
      } else if (key === 'promptTokens') {
        promptTokens = data[key];
      } else if (key === 'sender') {
        sender = data[key];
      } else if (!conversationId && key === 'conversationId') {
        conversationId = data[key];
      }
    }
  };

  const performCleanup = () => {
    logger.debug('[AgentController] Performing cleanup');
    if (Array.isArray(cleanupHandlers)) {
      for (const handler of cleanupHandlers) {
        try { if (typeof handler === 'function') handler(); }
        catch (e) { logger.error('[AgentController] Error in cleanup handler', e); }
      }
    }
    if (abortKey) {
      logger.debug('[AgentController] Cleaning up abort controller');
      cleanupAbortController(abortKey);
    }
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
    logger.debug('[AgentController] Cleanup completed');
  };

  try {
    const result = await initializeClient({ req, res, endpointOption });
    client = result.client;

    if (clientRegistry) clientRegistry.register(client, { userId }, client);
    requestDataMap.set(req, { client });

    const contentRef = new WeakRef(client.contentParts || []);
    getAbortData = () => {
      const content = contentRef.deref();
      return {
        sender,
        content: content || [],
        userMessage,
        promptTokens,
        conversationId,
        userMessagePromise,
        messageId: responseMessageId,
        parentMessageId: overrideParentMessageId ?? userMessageId,
      };
    };

    const { abortController, onStart } = createAbortController(req, res, getAbortData, getReqData);
    const closeHandler = () => {
      logger.debug('[AgentController] Request closed');
      if (!abortController || abortController.signal.aborted || abortController.requestCompleted) return;
      abortController.abort();
      logger.debug('[AgentController] Request aborted on close');
    };
    res.on('close', closeHandler);
    cleanupHandlers.push(() => {
      try { res.removeListener('close', closeHandler); }
      catch (e) { logger.error('[AgentController] Error removing close listener', e); }
    });

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

    let modelInput = text;

    try {
      logger.info(`[RAG] beginning retrieval for query: ${text}`);
      const ragResults = await getRagContext({ query: text });

      // Only show verbose RAG previews if DEBUG is enabled
      // Toggle raw RAG preview logs with env vars:
      //   LOG_LEVEL=debug and RAG_DEBUG=true
      if (DEBUG) {
        try {
          logger.debug(
            '[RAG] 🧪 Raw result previews:\n' +
            JSON.stringify(ragResults.slice(0, 8).map(r => ({
              sim: r.similarity?.toFixed(3),
              preview: (r.pageContent || '').slice(0, 120),
              source: r.metadata?.source || 'unknown',
            })), null, 2)
          );
        } catch (e) {
          logger.warn('[RAG] Failed to stringify preview:', e);
        }
      }

      const ragPrompt = formatRagContext(ragResults);
      if (ragPrompt?.length) {
        logger.info(`[RAG] ✅ Injecting RAG context of length: ${ragPrompt.length}`);
        modelInput = `${ragPrompt}\n\n${text}`;
      } else {
        logger.info(`[RAG] ❌ No RAG context injected`);
      }
    } catch (err) {
      logger.error('[AgentController] RAG injection failed:', err);
    }

    messageOptions.overrideParentMessageId = userMessageId;
    messageOptions.editedContent = text;

    let response = await client.sendMessage(text, {
      ...messageOptions,
      forcePrompt: modelInput,
    });

    // Normalize raw string
    if (typeof response === 'string') {
      logger.warn('[AgentController] Model returned raw string, normalizing');
      response = { messageId: `msg_${Date.now()}`, content: response };
    }

    // Validate shape
    if (!response || typeof response !== 'object') {
      logger.error('[AgentController] Malformed response from model:', `${response}`);
      return res.status(500).json({ error: 'Model returned malformed data' });
    }

    // Normalize text from content early
    if (!response.text && Array.isArray(response.content)) {
      try {
        response.text = response.content.map(c => (c && typeof c === 'object' ? (c.text || '') : String(c || ''))).join('\n');
      } catch (e) {
        logger.warn('[AgentController] Failed to derive text from content array:', e);
      }
    }

    const messageId = response.messageId;
    response.endpoint = endpointOption.endpoint;

    const databasePromise = response.databasePromise;
    delete response.databasePromise;

    const { conversation: convoData = {} } = await databasePromise;
    const conversation = { ...convoData };
    conversation.title = conversation?.title || 'New Chat';

    if (req.body.files && client.options?.attachments) {
      userMessage.files = [];
      const messageFiles = new Set(req.body.files.map(f => f.file_id));
      for (let att of client.options.attachments) {
        if (messageFiles.has(att.file_id)) {
          userMessage.files.push({ ...att });
        }
      }
      delete userMessage.image_urls;
    }

    if (!abortController.signal.aborted) {
      if (userMessage) {
        userMessage.content = text;
        userMessage.text = text;
      }

      logger.info('[AgentController] --- MODEL RESPONSE DIAGNOSTICS ---');

      if (typeof response === 'undefined') {
        logger.error('[AgentController] ❌ Response is UNDEFINED');
      } else if (response === null) {
        logger.error('[AgentController] ❌ Response is NULL');
      } else if (typeof response !== 'object') {
        logger.error(`[AgentController] ❌ Response is not an object: typeof = ${typeof response}`);
      } else {
        logger.info('[AgentController] ✅ Response is an object');
        const keys = Object.keys(response);
        logger.info(`[AgentController] Keys in model response: ${keys.join(', ')}`);

        if (keys.length === 0) {
          logger.warn('[AgentController] ⚠️ Response object has NO KEYS');
        }

        const safePreview = JSON.stringify(response, null, 2).slice(0, 2000);
        logger.info(`[AgentController] Preview of raw response:\n${safePreview}`);
      }

      const finalResponse = { ...response };

      // Sanitize & coerce array-like fields
      const bannedKeys = ['ragPrompt', 'ragContext', 'requestPrompt'];
      for (const key of bannedKeys) {
        if (key in finalResponse) delete finalResponse[key];
        if (userMessage && key in userMessage) delete userMessage[key];
      }
      const arrayFields = ['attachments','files','content','retrievedDocs','citations','tool_calls','sources'];
      for (const k of arrayFields) {
        if (k in finalResponse && !Array.isArray(finalResponse[k])) {
          logger.warn(`[Sanitizer] Coercing ${k} to [] from type ${typeof finalResponse[k]}`);
          finalResponse[k] = [];
        }
      }

      if (!finalResponse.text || typeof finalResponse.text !== 'string') {
        if (Array.isArray(finalResponse.content) && finalResponse.content.length > 0) {
          finalResponse.text = finalResponse.content.map(c => (c && typeof c === 'object' ? (c.text || '') : String(c || ''))).join('\n');
        }
      }

      if (!finalResponse.text || typeof finalResponse.text !== 'string') {
        logger.error('[AgentController] ❌ finalResponse.text is missing/invalid after normalization');
        try {
          const preview = JSON.stringify(finalResponse, null, 2).slice(0, 2000);
          logger.error(`[AgentController] finalResponse preview:\n${preview}`);
        } catch (e) {
          logger.error(`[AgentController] finalResponse could not be stringified: ${e?.message}`);
        }
        return res.status(500).json({ error: 'Invalid model response: missing text' });
      }

      try {
        const keys = Object.keys(finalResponse);
        const preview = JSON.stringify(finalResponse, null, 2).slice(0, 2000);
        logger.info(`[AgentController] ✅ Final payload being sent via sendEvent`);
        logger.info(`[AgentController] Keys in finalResponse: ${keys.join(', ')}`);
        logger.info(`[AgentController] typeof finalResponse: ${typeof finalResponse}`);
        logger.info(`[AgentController] Preview of finalResponse:\n${preview}`);
      } catch (e) {
        logger.warn(`[AgentController] Diagnostics failed: ${e?.message}`);
      }

      const ensureArray = (obj, key) => {
        if (!obj) return;
        if (!Array.isArray(obj[key])) obj[key] = [];
      };

      const extraArrayFields = [
        'files', 'attachments', 'content', 'retrievedDocs',
        'citations', 'tool_calls', 'sources', 'actions', 'moderations', 'spans',
      ];

      for (const k of extraArrayFields) {
        ensureArray(finalResponse, k);
        ensureArray(userMessage, k);
      }

      sendEvent(res, {
        final: true,
        conversation,
        title: conversation.title,
        requestMessage: userMessage,
        responseMessage: finalResponse,
      });
      res.end();

      if (client.savedMessageIds && !client.savedMessageIds.has(messageId)) {
        await saveMessage(
          req,
          { ...finalResponse, user: userId },
          { context: 'api/server/controllers/agents/request.js - response end' },
        );
      }
    } else if (!res.headersSent && !res.finished) {
      logger.debug('[AgentController] Handling edge case: aborted during sendCompletion');
      if (userMessage) {
        userMessage.content = text;
        userMessage.text = text;
      }
      const finalResponse = { ...response, error: true };
      sendEvent(res, {
        final: true,
        conversation,
        title: conversation.title,
        requestMessage: userMessage,
        responseMessage: finalResponse,
        error: { message: 'Request was aborted during completion' },
      });
      res.end();
    }
  } catch (error) {
    handleAbortError(res, req, error, {
      conversationId,
      sender,
      messageId: responseMessageId,
      parentMessageId: overrideParentMessageId ?? userMessageId ?? parentMessageId,
      userMessageId,
    })
      .catch((err) => {
        logger.error('[api/server/controllers/agents/request] Error in `handleAbortError`', err);
      })
      .finally(() => {
        performCleanup();
      });
  }
};

module.exports = AgentController;
