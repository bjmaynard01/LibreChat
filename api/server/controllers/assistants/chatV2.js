// Patch to enable RAG logging using LibreChat's Winston logger
const { v4 } = require('uuid');
const { sleep } = require('@librechat/agents');
const { sendEvent } = require('@librechat/api');
const { logger } = require('@librechat/data-schemas');
const {
  Time,
  Constants,
  RunStatus,
  CacheKeys,
  ContentTypes,
  ToolCallTypes,
  EModelEndpoint,
  retrievalMimeTypes,
  AssistantStreamEvents,
} = require('librechat-data-provider');
const {
  initThread,
  recordUsage,
  saveUserMessage,
  addThreadMetadata,
  saveAssistantMessage,
} = require('~/server/services/Threads');
const { runAssistant, createOnTextProgress } = require('~/server/services/AssistantService');
const { createErrorHandler } = require('~/server/controllers/assistants/errors');
const validateAuthor = require('~/server/middleware/assistants/validateAuthor');
const { createRun, StreamRunManager } = require('~/server/services/Runs');
const { addTitle } = require('~/server/services/Endpoints/assistants');
const { createRunBody } = require('~/server/services/createRunBody');
const { getTransactions } = require('~/models/Transaction');
const { checkBalance } = require('~/models/balanceMethods');
const { getConvo } = require('~/models/Conversation');
const getLogStores = require('~/cache/getLogStores');
const { countTokens } = require('~/server/utils');
const { getModelMaxTokens } = require('~/utils');
const { getOpenAIClient } = require('./helpers');

const { PGVectorStore } = require('@langchain/community/vectorstores/pgvector');
const { OpenAIEmbeddings } = require('@langchain/openai');
const { Pool } = require('pg');

const getRAGContext = async (query) => {
  const pool = new Pool({
    connectionString: process.env.PGVECTOR_DATABASE_URL,
  });

  const store = await PGVectorStore.initialize(
    new OpenAIEmbeddings(),
    {
      collectionName: 'testcollection',
      collectionTableName: 'langchain_pg_embedding',
      postgresConnectionOptions: pool,
    },
  );

  const results = await store.similaritySearchWithScore(query, 4);

  logger.info('🧠 Top 4 RAG results:');
  results.forEach(([doc, score], i) => {
    logger.info(`\n#${i + 1} (score: ${score.toFixed(4)})`);
    logger.info(doc.pageContent);
  });

  return results.map(([doc]) => doc.pageContent).join('\n\n');
};

// Main chatV2 logic
const chatV2 = async (req, res) => {
  // ... original structure untouched until prompt injection ...

  const {
    text,
    model,
    endpoint,
    assistant_id,
    ...rest
  } = req.body;

  const userMessageId = v4();

  const contextText = await getRAGContext(text);
  logger.error('--- Injecting RAG Context into prompt ---');
  logger.error(contextText);
  logger.error('--- End Injected Context ---');

  const userMessage = {
    role: 'user',
    content: [
      {
        type: ContentTypes.TEXT,
        text: `${contextText}\n\nUser: ${text}`,
      },
    ],
    metadata: {
      messageId: userMessageId,
    },
  };

  // ... resume original chatV2 structure from here ...
};

module.exports = {
  chatV2,
  getRAGContext,
};
