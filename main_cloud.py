from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import logging
import json

#Imports for GCP + embeddings 
from google.cloud import bigquery
from sentence_transformers import SentenceTransformer
from vertexai import init as vertex_init
from vertexai.generative_models import GenerativeModel
import google.cloud.logging
from pythonjsonlogger import jsonlogger

# Config 
PROJECT = "rag-demo-123456"
DATASET = "rag_demo"
TABLE = "faq_embeddings"
LOCATION = "us-central1"

#  Init Clients 
bq_client = bigquery.Client(project=PROJECT)
vertex_init(project=PROJECT, location=LOCATION)
gen_model = GenerativeModel("gemini-2.5-pro")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Setup structured logging 
log_client = google.cloud.logging.Client()
log_client.setup_logging()

logger = logging.getLogger("uvicorn")
handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
handler.setFormatter(formatter)
logger.handlers = [handler]
logger.setLevel(logging.INFO)

# FastAPI + UI templates 
templates = Jinja2Templates(directory="templates")
app = FastAPI(title="RAG on GCP", description="BigQuery retrieval + Gemini answering")


# Retrieval 
def retrieve_topk(question: str, k: int = 3):
    q_emb = embedder.encode(question).tolist()
    sql = f"""
    SELECT question, answer,
           ML.DISTANCE(embedding, {q_emb}, 'COSINE') AS similarity
    FROM `{PROJECT}.{DATASET}.{TABLE}`
    ORDER BY similarity ASC
    LIMIT {k}
    """
    rows = bq_client.query(sql).result()
    return [(r.question, r.answer) for r in rows]


#  Generation
def generate_answer(question: str, answers: list[str]) -> str:
    context = "\n\n".join(answers)
    prompt = f"""You are a precise assistant. Use ONLY the context to answer.

Context:
{context}

Question: {question}

If the answer is not in the context, say you don't know based on the provided context.
"""
    response = gen_model.generate_content(prompt)
    return response.text


#  Routes 
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/query", response_class=HTMLResponse)
def query(request: Request, question: str = Query(..., description="Your question")):
    try:
        docs = retrieve_topk(question, k=3)
        support_answers = [a for (_, a) in docs]
        final = generate_answer(question, support_answers)

        #  Log query metric
        logger.info({
            "event": "metric",
            "metric_type": "queries_total",
            "count": 1,
            "question": question
        })

        return templates.TemplateResponse("result.html", {
            "request": request,
            "question": question,
            "answer": final,
            "docs": [q for (q, _) in docs],
            "answers": support_answers
        })

    except Exception as e:
        #  Log error metric
        logger.error({
            "event": "metric",
            "metric_type": "errors_total",
            "count": 1,
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail="Internal server error")
