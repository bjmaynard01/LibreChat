import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";

const embedder = new OllamaEmbeddings({
  model: "nomic-embed-text:latest",
  baseUrl: process.env.OLLAMA_BASE_URL,
});

const store = await PGVectorStore.initialize(embedder, {
  collectionName: "global_docs",
  collectionTableName: "langchain_pg_collection",
  postgresConnectionOptions: { connectionString: process.env.DATABASE_URL },
  distanceStrategy: "cosine",
});

const query = process.argv.slice(2).join(" ") || "VA Form 21-526EZ";
const k = 8;
const hits = await store.similaritySearchWithScore(query, k);

console.log(
  hits.map(([d, dist]) => ({
    dist,
    sim: 1 - dist,
    text: d.pageContent.slice(0, 140),
    meta: d.metadata,
  }))
);
