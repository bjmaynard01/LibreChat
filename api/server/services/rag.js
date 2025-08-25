// api/server/services/rag.js
const { logger } = require('@librechat/data-schemas');
const { PGVectorStore } = require('@langchain/community/vectorstores/pgvector');
const { OllamaEmbeddings } = require('@langchain/community/embeddings/ollama');

const RAG_K = parseInt(process.env.RAG_K || '8', 10);
const MAX_COSINE_DIST = Number(process.env.RAG_MAX_COSINE_DIST || 0.45);

async function getRagContext({ query }) {
  logger.info(`🚀 [RAG] Starting retrieval for query: "${query}"`);

  try {
    const embedder = new OllamaEmbeddings({
      model: process.env.EMBED_MODEL || 'nomic-embed-text:latest',
      baseUrl: process.env.OLLAMA_BASE_URL,
    });

    const store = await PGVectorStore.initialize(embedder, {
      collectionName: process.env.VECTOR_COLLECTION || 'global_docs',
      collectionTableName: process.env.VECTOR_COLLECTION_TABLE || 'langchain_pg_collection',
      collectionTableRowId: process.env.VECTOR_COLLECTION_PK || 'uuid',
      tableName: process.env.VECTOR_TABLE || 'langchain_pg_embedding',
      columns: {
        idColumnName: process.env.VECTOR_ID_COLUMN || 'uuid',
        vectorColumnName: process.env.VECTOR_VEC_COLUMN || 'embedding',
        contentColumnName: process.env.VECTOR_CONTENT_COLUMN || 'document',
        metadataColumnName: process.env.VECTOR_META_COLUMN || 'cmetadata',
      },
      postgresConnectionOptions: {
        connectionString: process.env.DATABASE_URL,
        max: 2,
        idleTimeoutMillis: 1000,
      },
      distanceStrategy: 'cosine',
    });

    const results = await store.similaritySearchWithScore(query, RAG_K);
    logger.info(`🔍 [RAG] similarity results: ${results.length}`);

    if (!results.length) {
      logger.info('[RAG] ❌ No vector matches returned.');
      return [];
    }

    const debugView = results.map(([doc, dist]) => ({
      sim: (1 - dist).toFixed(3),
      preview: doc.pageContent.slice(0, 140),
      source: doc.metadata?.source || 'unknown',
    }));
    logger.info(`[RAG] 🧪 Raw result previews:\n${JSON.stringify(debugView, null, 2)}`);

    const kept = results.filter(([, dist]) => dist <= MAX_COSINE_DIST);
    if (!kept.length) {
      logger.info(`[RAG] 📉 No chunks passed cutoff of ${MAX_COSINE_DIST}.`);
      return [];
    }

    const simScores = kept.map(([_, dist]) => (1 - dist).toFixed(3));
    logger.info(`[RAG] ✅ Injecting ${kept.length} chunks with similarity scores: ${simScores.join(', ')}`);

    return kept.map(([doc, dist]) => ({
      dist,
      sim: 1 - dist,
      text: doc.pageContent,
      meta: doc.metadata || {},
    }));
  } catch (e) {
    logger.error(`[RAG] ❗️ Retrieval failed: ${e?.message || e}`);
    if (e?.stack) logger.error(e.stack);
    return [];
  }
}


function formatRagContext(docs) {
  if (!docs?.length) return '';
  const blocks = docs.map(({ text, meta }) => {
    const title = meta.title || meta.section_path || meta.source || 'Source';
    const url = meta.url ? ` (${meta.url})` : '';
    return `• ${title}${url}\n${text}`;
  });
  const joined = blocks.join('\n\n');
  return joined.length > 8000 ? joined.slice(0, 8000) + '\n…' : joined;
}


module.exports = { getRagContext, formatRagContext };
