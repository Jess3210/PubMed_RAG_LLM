from typing import List
from fastapi import FastAPI, Request
import numpy as np
from pydantic import BaseModel
from classes.AnswerLLMClass import AnswerLLM
from classes.DatabaseChromaDBClass import DatabaseChromaDB
from classes.ExtractPubMedDataClass import ExtractPubMedData
from classes.SimilarityMetricClass import SimilarityMetric
from contextlib import asynccontextmanager

# Request body for /ingest endpoint
class IngestRequest(BaseModel):
    document: str  # The input document to embed and store

# Request body for /query endpoint
class QueryRequest(BaseModel):
    query: str  # The user's search query

def run_insertion_pipeline(input_urls:List[str]):
    """
    Run complete insertion pipeline and insert the paper into the ChromaDB

    Parameters:
        input_urls (List[str]): input paper url
    
    Returns:
        None
    """
    #Extract data, embedding and transform it with metadata
    extract_data = ExtractPubMedData(input_urls)
    extract_data.extract_pubmed_data()

    #Insert into database ChromaDB
    database = DatabaseChromaDB()
    database.insert(extract_data.paper_data)

def run_query_pipeline(query:str, 
                       credentials:str = "CREDENTIALS.json", 
                       project_id:str = "PROJECT_NAME"):
    """
    Run complete query pipeline to ask a question and get the answer based on retrieved context.

    Parameters:
        query (str): Question query
        credentials (str): GCP credentials
        project_id (str): GCP project ID
    
    Returns:
        None
    """
    #Get data from database ChromaDB
    database = DatabaseChromaDB()
    res = database.query_collection()

    question = {'abstract': [query]}

    #embedding of query
    extract_data = ExtractPubMedData()
    embedded_question = np.array(extract_data.embedding(question)['embedding'][0])

    #Cosine similarity and ranking
    sim_rank = SimilarityMetric(embedded_question, res)
    ranking_res = sim_rank.ranking()

    #Generate answer
    answer_now = AnswerLLM(credentials=credentials, project_id=project_id)
    answer = answer_now.generate_answer(question["abstract"][0], res['documents'][ranking_res[0]])
    return(answer, res['uris'][ranking_res[0]])

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for initial data insertion at the start of FastAPI-Anwendung.
    """

    # PubMed URL
    pubmed_urls = ["https://pubmed.ncbi.nlm.nih.gov/15858239/", "https://pubmed.ncbi.nlm.nih.gov/20598273/", "https://pubmed.ncbi.nlm.nih.gov/6650562/"]
    run_insertion_pipeline(pubmed_urls)
    
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/ingest/")
def ingest(req: IngestRequest) -> dict:
    """
    Ingest a document and store its embedding in the database.
    """
    run_insertion_pipeline([req.document])
    return {"status": "ok"}

@app.post("/query")
def query(req: QueryRequest) -> dict:
    """
    Search for relevant documents and generate an answer based on the query.
    """
    result = run_query_pipeline(req.query)
    
    #return the question, generated answer and the url to the original paper
    return {"question:" : req.query,
            "answer": result[0],
            "url" : result[1]}

