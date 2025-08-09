// sql_check.js
const { Client } = require("pg");

(async () => {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();

  const r1 = await client.query("SELECT current_database() db, current_user usr, current_schema() sch");
  console.log("🔌 DB ctx:", r1.rows[0]);

  const r2 = await client.query("SELECT name, uuid FROM langchain_pg_collection");
  console.log("📚 Collections:", r2.rows);

  const r3 = await client.query(`
    SELECT c.name, COUNT(*) AS rows
    FROM langchain_pg_embedding e
    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
    GROUP BY c.name
    ORDER BY rows DESC
  `);
  console.log("📦 Row counts:", r3.rows);

  const r4 = await client.query(`
    SELECT LEFT(e.document, 200) AS snippet
    FROM langchain_pg_embedding e
    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
    WHERE c.name = 'global_docs'
    LIMIT 2
  `);
  console.log("📝 Sample:", r4.rows);

  await client.end();
})();
