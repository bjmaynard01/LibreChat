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

    return context;
  } catch (e) {
    logger.error(`❗️Error in getRAGContext(): ${e?.message || e}`);
    if (e?.stack) logger.error(e.stack);
    return '';
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
