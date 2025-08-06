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
const { OllamaEmbeddings } = require('@langchain/community/embeddings/ollama');
const { Pool } = require('pg');

const RAG_SCORE_THRESHOLD = 0.75;

const getRAGContext = async (query) => {
  logger.debug('Starting getRAGContext()');
  try {
    const store = await PGVectorStore.initialize(
      new OllamaEmbeddings({
        model: 'nomic-embed-text:latest',
        baseUrl: process.env.OLLAMA_BASE_URL,
      }),
      {
        collectionName: 'testcollection',
        collectionTableName: 'langchain_pg_collection',
        postgresConnectionOptions: {
          connectionString: process.env.DATABASE_URL,
          max: 2,
          idleTimeoutMillis: 1000,
        },
      }
    );
    logger.debug('Vector store initialized');

    const results = await store.similaritySearchWithScore(query, 4);

    if (!results?.length) {
      logger.info('No RAG results found.');
      return '';
    }

    const filtered = results.filter(([_, score]) => score >= RAG_SCORE_THRESHOLD);

    if (!filtered.length) {
      logger.info('RAG results all below threshold — skipping context injection.');
      return '';
    }

    logger.info(`Filtered ${filtered.length} high-confidence RAG chunks:`);
    filtered.forEach(([doc, score], i) => {
      logger.info(`Result #${i + 1} (score: ${score.toFixed(4)})`);
      logger.info(`Page content:\n${doc?.pageContent || '[No content]'}`);
    });

    return filtered.map(([doc]) => doc.pageContent).join('\n\n');
  } catch (e) {
    logger.error(`Error in getRAGContext(): ${e?.message || e}`);
    if (e?.stack) {
      logger.error(e.stack);
    }
    return '';
  }
};

const chatV2 = async (req, res) => {
  const {
    text,
    model,
    endpoint,
    assistant_id,
    ...rest
  } = req.body;

  const userMessageId = v4();
  const contextText = await getRAGContext(text);

  if (contextText) {
    logger.info('Injecting RAG context into prompt');
  } else {
    logger.info('No RAG context injected for this query');
  }

  const userMessage = {
    role: 'user',
    content: [
      {
        type: ContentTypes.TEXT,
        text: `${contextText ? contextText + '\n\n' : ''}User: ${text}`,
      },
    ],
    metadata: {
      messageId: userMessageId,
    },
  };

  // TODO: Continue processing the message as LibreChat normally does.
  // This file defines only RAG context injection; the rest of the logic remains unchanged.
};

module.exports = {
  chatV2,
  getRAGContext,
};
