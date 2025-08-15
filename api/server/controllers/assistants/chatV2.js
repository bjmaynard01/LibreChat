// api/server/controllers/assistants/chatV2.js

const { v4 } = require('uuid');
const { logger } = require('@librechat/data-schemas');
const { ContentTypes } = require('librechat-data-provider');

const { PGVectorStore } = require('@langchain/community/vectorstores/pgvector');
const { OllamaEmbeddings } = require('@langchain/community/embeddings/ollama');

const RAG_K = 8;
const MAX_COSINE_DIST = 0.45;

const getRAGContext = async (query) => {
  logger.info('🚀 Starting getRAGContext()');

  try {
    const embedder = new OllamaEmbeddings({
      model: 'nomic-embed-text:latest',
      baseUrl: process.env.OLLAMA_BASE_URL,
    });

    const store = await PGVectorStore.initialize(embedder, {
      collectionName: 'global_docs',
      collectionTableName: 'langchain_pg_collection',
      collectionTableRowId: 'uuid',
      tableName: 'langchain_pg_embedding',
      columns: {
        idColumnName: 'uuid',
        vectorColumnName: 'embedding',
        contentColumnName: 'document',
        metadataColumnName: 'cmetadata',
      },
      postgresConnectionOptions: {
        connectionString: process.env.DATABASE_URL,
        max: 2,
        idleTimeoutMillis: 1000,
      },
      distanceStrategy: 'cosine',
    });

    const results = await store.similaritySearchWithScore(query, RAG_K);
    logger.info(`🔍 RAG similarity results count (raw): ${results.length}`);

    if (!results.length) {
      logger.info('❌ No RAG matches found.');
      return '';
    }

    const debugView = results.map(([doc, dist]) => ({
      dist,
      sim: 1 - dist,
      preview: doc?.pageContent?.slice(0, 140),
      source: doc?.metadata?.source,
    }));
    logger.info(`🧪 Raw hits:\n${JSON.stringify(debugView, null, 2)}`);

    const kept = results.filter(([_, dist]) => dist <= MAX_COSINE_DIST);
    if (!kept.length) {
      logger.info(
        `📉 No hits passed MAX_COSINE_DIST=${MAX_COSINE_DIST}. Try raising threshold to confirm flow.`
      );
      return '';
    }

    logger.info(`✅ Kept ${kept.length}/${results.length} chunk(s)`);

    const context = kept.map(([doc]) => doc.pageContent).join('\n\n');
    logger.info(`🧩 Final injected RAG context:\n${context.slice(0, 3000)}...`);

    const getRequestFileIds = async () => {
      let thread_file_ids = [];
      if (convoId) {
        const convo = await getConvo(req.user.id, convoId);
        if (convo && convo.file_ids) {
          thread_file_ids = convo.file_ids;
        }
      }

      if (files.length || thread_file_ids.length) {
        attachedFileIds = new Set([...file_ids, ...thread_file_ids]);

        let attachmentIndex = 0;
        for (const file of files) {
          file_ids.push(file.file_id);
          if (file.type.startsWith('image')) {
            userMessage.content.push({
              type: ContentTypes.IMAGE_FILE,
              [ContentTypes.IMAGE_FILE]: { file_id: file.file_id },
            });
          }

          if (!userMessage.attachments) {
            userMessage.attachments = [];
          }

          userMessage.attachments.push({
            file_id: file.file_id,
            tools: [{ type: ToolCallTypes.CODE_INTERPRETER }],
          });

          if (file.type.startsWith('image')) {
            continue;
          }

          const mimeType = file.type;
          const isSupportedByRetrieval = retrievalMimeTypes.some((regex) => regex.test(mimeType));
          if (isSupportedByRetrieval) {
            userMessage.attachments[attachmentIndex].tools.push({
              type: ToolCallTypes.FILE_SEARCH,
            });
          }

          attachmentIndex++;
        }
      }
    };

    /** @type {Promise<Run>|undefined} */
    let userMessagePromise;

    const initializeThread = async () => {
      await getRequestFileIds();

      // TODO: may allow multiple messages to be created beforehand in a future update
      const initThreadBody = {
        messages: [userMessage],
        metadata: {
          user: req.user.id,
          conversationId,
        },
      };

      const result = await initThread({ openai, body: initThreadBody, thread_id });
      thread_id = result.thread_id;

      createOnTextProgress({
        openai,
        conversationId,
        userMessageId,
        messageId: responseMessageId,
        thread_id,
      });

      requestMessage = {
        user: req.user.id,
        text,
        messageId: userMessageId,
        parentMessageId,
        // TODO: make sure client sends correct format for `files`, use zod
        files,
        file_ids,
        conversationId,
        isCreatedByUser: true,
        assistant_id,
        thread_id,
        model: assistant_id,
        endpoint,
      };

      previousMessages.push(requestMessage);

      /* asynchronous */
      userMessagePromise = saveUserMessage(req, { ...requestMessage, model });

      conversation = {
        conversationId,
        endpoint,
        promptPrefix: promptPrefix,
        instructions: instructions,
        assistant_id,
        // model,
      };

      if (file_ids.length) {
        conversation.file_ids = file_ids;
      }
    };

    const promises = [initializeThread(), checkBalanceBeforeRun()];
    await Promise.all(promises);

    const sendInitialResponse = () => {
      sendEvent(res, {
        sync: true,
        conversationId,
        // messages: previousMessages,
        requestMessage,
        responseMessage: {
          user: req.user.id,
          messageId: openai.responseMessage.messageId,
          parentMessageId: userMessageId,
          conversationId,
          assistant_id,
          thread_id,
          model: assistant_id,
        },
      });
    };

    /** @type {RunResponse | typeof StreamRunManager | undefined} */
    let response;

    const processRun = async (retry = false) => {
      if (endpoint === EModelEndpoint.azureAssistants) {
        body.model = openai._options.model;
        openai.attachedFileIds = attachedFileIds;
        if (retry) {
          response = await runAssistant({
            openai,
            thread_id,
            run_id,
            in_progress: openai.in_progress,
          });
          return;
        }

        /* NOTE:
         * By default, a Run will use the model and tools configuration specified in Assistant object,
         * but you can override most of these when creating the Run for added flexibility:
         */
        const run = await createRun({
          openai,
          thread_id,
          body,
        });

        run_id = run.id;
        await cache.set(cacheKey, `${thread_id}:${run_id}`, Time.TEN_MINUTES);
        sendInitialResponse();

        // todo: retry logic
        response = await runAssistant({ openai, thread_id, run_id });
        return;
      }

      /** @type {{[AssistantStreamEvents.ThreadRunCreated]: (event: ThreadRunCreated) => Promise<void>}} */
      const handlers = {
        [AssistantStreamEvents.ThreadRunCreated]: async (event) => {
          await cache.set(cacheKey, `${thread_id}:${event.data.id}`, Time.TEN_MINUTES);
          run_id = event.data.id;
          sendInitialResponse();
        },
      };

      /** @type {undefined | TAssistantEndpoint} */
      const config = req.app.locals[endpoint] ?? {};
      /** @type {undefined | TBaseEndpoint} */
      const allConfig = req.app.locals.all;

      const streamRunManager = new StreamRunManager({
        req,
        res,
        openai,
        handlers,
        thread_id,
        attachedFileIds,
        parentMessageId: userMessageId,
        responseMessage: openai.responseMessage,
        streamRate: allConfig?.streamRate ?? config.streamRate,
        // streamOptions: {

        // },
      });

      await streamRunManager.runAssistant({
        thread_id,
        body,
      });

      response = streamRunManager;
      response.text = streamRunManager.intermediateText;
    };

    await processRun();
    logger.debug('[/assistants/chat/] response', {
      run: response.run,
      steps: response.steps,
    });

    if (response.run.status === RunStatus.CANCELLED) {
      logger.debug('[/assistants/chat/] Run cancelled, handled by `abortRun`');
      return res.end();
    }

    if (response.run.status === RunStatus.IN_PROGRESS) {
      processRun(true);
    }

    completedRun = response.run;

    /** @type {ResponseMessage} */
    const responseMessage = {
      ...(response.responseMessage ?? response.finalMessage),
      text: response.text,
      parentMessageId: userMessageId,
      conversationId,
      user: req.user.id,
      assistant_id,
      thread_id,
      model: assistant_id,
      endpoint,
      spec: endpointOption.spec,
      iconURL: endpointOption.iconURL,
    };

    sendEvent(res, {
      final: true,
      conversation,
      requestMessage: {
        parentMessageId,
        thread_id,
      },
    });
    res.end();

    if (userMessagePromise) {
      await userMessagePromise;
    }
    await saveAssistantMessage(req, { ...responseMessage, model });

    if (parentMessageId === Constants.NO_PARENT && !_thread_id) {
      addTitle(req, {
        text,
        responseText: response.text,
        conversationId,
        client,
      });
    }

    await addThreadMetadata({
      openai,
      thread_id,
      messageId: responseMessage.messageId,
      messages: response.messages,
    });

    if (!response.run.usage) {
      await sleep(3000);
      completedRun = await openai.beta.threads.runs.retrieve(response.run.id, { thread_id });
      if (completedRun.usage) {
        await recordUsage({
          ...completedRun.usage,
          user: req.user.id,
          model: completedRun.model ?? model,
          conversationId,
        });
      }
    } else {
      await recordUsage({
        ...response.run.usage,
        user: req.user.id,
        model: response.run.model ?? model,
        conversationId,
      });
    }
  } catch (error) {
    await handleError(error);
  }
};

const chatV2 = async (req, res) => {
  const { text, model, endpoint, assistant_id, ...rest } = req.body;
  const userMessageId = v4();

  const contextText = await getRAGContext(text);

  // User-visible message
  const userMessage = {
    role: 'user',
    content: [
      {
        type: ContentTypes.TEXT,
        text: text, // Show just the user input in the UI
      },
    ],
    metadata: {
      messageId: userMessageId,
      contextInjected: !!contextText,
    },
  };

  const systemMessage = contextText
    ? {
        role: 'system',
        content: [
          {
            type: ContentTypes.TEXT,
            text:
              'Use the following context to answer the user’s question. If the context is insufficient, say so.\n\n' +
              contextText,
          },
        ],
      }
    : null;

  // 👇 Inject messages into prompt
  req.body.messages = systemMessage
    ? [systemMessage, userMessage]
    : [userMessage];

  req.body.userMessageId = userMessageId;

  return res.locals.chat(req, res);
};

module.exports = { chatV2, getRAGContext };
