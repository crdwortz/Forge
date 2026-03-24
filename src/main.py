"""
FastAPI application for Wikipedia RAG system.

This application provides endpoints for:
1. Data ingestion: Loading Wikipedia passages and creating embeddings
2. Query/Retrieval: Embedding user queries and searching the vector database
3. Answer Generation: Augmenting prompts and generating answers with the LLM
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging

try:
    from .ingestion import DataIngestionService
    from .retrieval import RetrievalService
    from .generation import GenerationService
except ImportError:
    from ingestion import DataIngestionService
    from retrieval import RetrievalService
    from generation import GenerationService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Wikipedia RAG System",
    description="A Retrieval-Augmented Generation system for Wikipedia articles",
    version="0.1.0"
)

# Initialize services
ingestion_service = DataIngestionService()
retrieval_service = RetrievalService(ingestion_service)
generation_service = GenerationService()


# ============================================================================
# Data Models
# ============================================================================

class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    message: str


class IngestRequest(BaseModel):
    """Request model for data ingestion."""
    max_documents: Optional[int] = None
    force_reload: bool = False


class IngestResponse(BaseModel):
    """Response model for ingestion endpoint."""
    status: str
    documents_loaded: int
    embeddings_created: int
    message: str


class DocumentMatch(BaseModel):
    """Model for a matched document."""
    content: str
    score: float
    metadata: Optional[dict] = None


class SearchRequest(BaseModel):
    """Request model for document search."""
    query: str
    top_k: int = 5


class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    query: str
    matches: List[DocumentMatch]
    count: int


class QueryRequest(BaseModel):
    """Request model for full RAG query."""
    query: str
    top_k: int = 5
    temperature: float = 0.7


class QueryResponse(BaseModel):
    """Response model for RAG query."""
    query: str
    answer: str
    sources: List[DocumentMatch]
    tokens_used: Optional[dict] = None


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Verify services are initialized
        if not ingestion_service or not retrieval_service or not generation_service:
            raise Exception("Services not properly initialized")
        
        return HealthResponse(
            status="healthy",
            message="Wiki RAG system is operational"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_data(request: IngestRequest):
    """
    Ingest Wikipedia passages, create embeddings, and populate vector database.
    
    This endpoint:
    1. Loads passages from HuggingFace dataset
    2. Creates embeddings using text-embedding-3-large
    3. Stores in local LlamaIndex vector database
    """
    try:
        logger.info(f"Starting data ingestion (max_documents={request.max_documents})")
        
        result = await ingestion_service.ingest_wikipedia_data(
            max_documents=request.max_documents,
            force_reload=request.force_reload
        )
        
        return IngestResponse(
            status="success",
            documents_loaded=result["documents_loaded"],
            embeddings_created=result["embeddings_created"],
            message=f"Successfully loaded {result['documents_loaded']} documents"
        )
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search the vector database for documents matching the query.
    
    Args:
        query: The search query
        top_k: Number of top matches to return (default: 5)
    """
    try:
        logger.info(f"Searching for: {request.query}")
        
        matches = await retrieval_service.retrieve_documents(
            query=request.query,
            top_k=request.top_k
        )
        
        return SearchResponse(
            query=request.query,
            matches=[
                DocumentMatch(
                    content=match["content"],
                    score=match["score"],
                    metadata=match.get("metadata")
                )
                for match in matches
            ],
            count=len(matches)
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def rag_query(request: QueryRequest):
    """
    Full RAG pipeline: retrieve documents and generate an answer.
    
    Args:
        query: The user's question
        top_k: Number of documents to retrieve (default: 5)
        temperature: LLM temperature for response generation (default: 0.7)
    """
    try:
        logger.info(f"Processing RAG query: {request.query}")
        
        # Step 1: Retrieve relevant documents
        documents = await retrieval_service.retrieve_documents(
            query=request.query,
            top_k=request.top_k
        )
        
        # Step 2: Generate answer using retrieved documents
        result = await generation_service.generate_answer(
            query=request.query,
            documents=documents,
            temperature=request.temperature
        )
        
        return QueryResponse(
            query=request.query,
            answer=result["answer"],
            sources=[
                DocumentMatch(
                    content=doc["content"],
                    score=doc["score"],
                    metadata=doc.get("metadata")
                )
                for doc in documents
            ],
            tokens_used=result.get("tokens_used")
        )
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
