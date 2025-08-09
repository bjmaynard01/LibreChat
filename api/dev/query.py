import psycopg2
import requests
import os
import json

# Load the query string you want to test RAG against
query_text = "What kinds of disability benefits are available to veterans?"

# 1. Generate the embedding via your embedding endpoint
EMBEDDING_ENDPOINT = os.environ.get("EMBEDDING_ENDPOINT", "http://ollama:11434/api/embeddings")
embedding_payload = {
    "model": "nomic-embed-text",
    "prompt": query_text
}

response = requests.post(EMBEDDING_ENDPOINT, json=embedding_payload)
response.raise_for_status()
try:
    embedding = response.json()["embedding"]
except KeyError:
    print("Unexpected response from Ollama:")
    print(response.text)
    raise

print(f"Query embedding dimension: {len(embedding)}")

# 2. Connect to the PGVector database
conn = psycopg2.connect("dbname=mydatabase user=myuser password=mypassword host=vectordb")
cur = conn.cursor()

# 3. Execute cosine similarity query against the vector store
embedding_str = f"ARRAY{embedding}::vector"
query = f"""
    SELECT
        uuid,
        custom_id,
        LEFT(document, 200) AS preview,
        embedding <#> {embedding_str} AS cosine_distance
    FROM langchain_pg_embedding
    WHERE collection_id = '6f840ac7-b242-4f31-b7d2-38e8eea0fed5'
    ORDER BY cosine_distance ASC
    LIMIT 5;
"""

cur.execute(query)
rows = cur.fetchall()

print(f"\nTop matches for query: \"{query_text}\"\n")
for row in rows:
    print(row)

cur.close()
conn.close()
