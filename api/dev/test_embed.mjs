import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";

const embedder = new OllamaEmbeddings({
  model: "nomic-embed-text",
  baseUrl: process.env.OLLAMA_BASE_URL,
});

const query = "How do I apply for VA disability compensation?";
const vec = await embedder.embedQuery(query);

console.log("Embedding length:", vec.length);
console.log("First 10 dims:", vec.slice(0, 10));
