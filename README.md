# RAG-Knowledge-assistant

This project is a Retrieval-Augmented Generation (RAG) system built on Google Cloud Platform (GCP). 
The idea is simple: Large Language Models (LLMs) are powerful, but they sometimes “hallucinate.”  
By combining them with trusted knowledge from BigQuery, we can make the answers accurate, grounded, and useful.  


## What This Project Does
- You type a question into a clean, dark-themed web page.  
- The system finds the most relevant answers from a knowledge base stored in BigQuery.  
- These results are passed to **Vertex AI Gemini**, which generates a final, grounded answer.  
- You instantly see the result along with supporting context.  
