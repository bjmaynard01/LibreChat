const { PGVectorStore } = require("langchain/vectorstores/pgvector");
const { OllamaEmbeddings } = require("langchain/embeddings/ollama");

(async () => {
  const store = await PGVectorStore.initialize(
    new OllamaEmbeddings({
      model: "nomic-embed-text:latest",
      baseUrl: "http://ollama:11434",
    }),
    {
      collectionName: "global_docs",
      collectionTableName: "langchain_pg_collection",
      postgresConnectionOptions: {
        connectionString: "postgresql://myuser:mypassword@vectordb:5432/mydatabase",
      },
    }
  );

  const results = await store.similaritySearchWithScore("veterans disability benefits", 5);
  for (const [doc, score] of results) {
    console.log(`Score: ${score.toFixed(4)} — ${doc.pageContent.slice(0, 120)}\n`);
  }
})();
