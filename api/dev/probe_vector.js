// probe_vector.js
const { PGVectorStore } = require("@langchain/community/vectorstores/pgvector");
const { OllamaEmbeddings } = require("@langchain/community/embeddings/ollama");

(async () => {
  const text = process.argv.slice(2).join(" ") || "VA Form 21-526EZ";

  const embedder = new OllamaEmbeddings({
    model: "nomic-embed-text:latest",
    baseUrl: process.env.OLLAMA_BASE_URL,
  });
  const vec = await embedder.embedQuery(text);

  const store = await PGVectorStore.initialize(embedder, {
    collectionName: "global_docs",
    collectionTableName: "langchain_pg_collection",
    collectionTableRowId: "uuid",
    postgresConnectionOptions: { connectionString: process.env.DATABASE_URL },
    distanceStrategy: "cosine",
  });

  // Use the vector API instead of text API
  const hits = await store.similaritySearchVectorWithScore(vec, 8);
  console.log(
    hits.map(([d, dist]) => ({
      dist,
      sim: 1 - dist,
      text: d.pageContent.slice(0, 140),
      meta: d.metadata,
    }))
  );
})();
