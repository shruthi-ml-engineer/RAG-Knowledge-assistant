import json
from sentence_transformers import SentenceTransformer
from google.cloud import bigquery


PROJECT = "rag-demo-123456"      # <-- replace with your actual GCP project id
DATASET = "rag_demo"             # dataset you created earlier with bq mk
TABLE   = "faq_embeddings"       # new table to create

#load your FAQ data
with open("C:/Shruthi_tasks/Important_documents/codes/github/RAG_LLM/faq.json", "r", encoding="utf-8") as f:
    faqs = json.load(f)

#embed using a small transformer model
print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

rows = []
for i, faq in enumerate(faqs):
    q = faq["question"].strip()

    # if answer is a list, join it into one string
    a = faq["answer"]
    if isinstance(a, list):
        a = " ".join(a)
    a = a.strip()

    emb = model.encode(q).tolist()
    rows.append({"id": i, "question": q, "answer": a, "embedding": emb})


#upload to BigQuery
print(f"Uploading {len(rows)} rows to BigQuery...")
client = bigquery.Client(project=PROJECT)
table_id = f"{PROJECT}.{DATASET}.{TABLE}"

schema = [
    bigquery.SchemaField("id", "INTEGER"),
    bigquery.SchemaField("question", "STRING"),
    bigquery.SchemaField("answer", "STRING"),
    bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
]

job = client.load_table_from_json(
    rows,
    table_id,
    job_config=bigquery.LoadJobConfig(
        schema=schema,
        write_disposition="WRITE_TRUNCATE"  # overwrite if table exists
    )
)
job.result()

print(f"Done! Data is in BigQuery table: {table_id}")
