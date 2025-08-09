// embed_sql_cosine.js
const { Client } = require("pg");
const { OllamaEmbeddings } = require("@langchain/community/embeddings/ollama");

(async () => {
  const query = process.argv.slice(2).join(" ") || "VA Form 21-526EZ";

  const emb = new OllamaEmbeddings({
    model: "nomic-embed-text:latest",
    baseUrl: process.env.OLLAMA_BASE_URL,
  });

  const vec = await emb.embedQuery(query);
  console.log("🧠 Query dims:", vec.length, "first5:", vec.slice(0,5));

  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();

  const sql = `
    WITH q AS (SELECT $1::vector AS v)
    SELECT LEFT(e.document, 160) AS snippet,
           1 - (e.embedding <=> (SELECT v FROM q)) AS cosine_sim
    FROM langchain_pg_embedding e
    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
    WHERE c.name = 'global_docs'
    ORDER BY e.embedding <=> (SELECT v FROM q)
    LIMIT 5
  `;
  const res = await client.query(sql, [JSON.stringify(vec)]);
  console.log("🔎 Top 5:", res.rows);

  await client.end();
})();
