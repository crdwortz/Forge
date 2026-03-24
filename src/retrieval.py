"""
Retrieval module for Wikipedia RAG system.

Handles embedding user queries and searching the vector database for relevant documents.
"""

import logging
from typing import List, Dict, Any, Optional

try:
    from .ingestion import DataIngestionService
    from .llamaindex_models import get_embedding_model
except ImportError:
    from ingestion import DataIngestionService
    from llamaindex_models import get_embedding_model

logger = logging.getLogger(__name__)


class RetrievalService:
    """Service for retrieving documents from the vector database."""
    
    def __init__(self):
        """Initialize the retrieval service."""
        self.ingestion_service = DataIngestionService()
        self.embedding_model = get_embedding_model()
    
    async def retrieve_documents(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant documents for a query.
        
        Args:
            query: The search query
            top_k: Number of top matches to return
            
        Returns:
            List of documents with content, score, and metadata
        """
        try:
            logger.info(f"Retrieving documents for query: {query}")
            
            # Get the vector index
            index = self.ingestion_service.get_index()
            
            # Create a retriever with specified top-k
            retriever = index.as_retriever(similarity_top_k=top_k)
            
            # Retrieve documents
            retrieved_nodes = retriever.retrieve(query)
            logger.info(f"Retrieved {len(retrieved_nodes)} documents")
            
            # Convert nodes to dictionary format
            results = []
            for node in retrieved_nodes:
                result = {
                    "content": node.get_content(),
                    "score": node.score if hasattr(node, 'score') else 0.0,
                    "metadata": node.metadata if hasattr(node, 'metadata') else {},
                    "doc_id": node.doc_id if hasattr(node, 'doc_id') else None
                }
                results.append(result)
                logger.debug(f"Document score: {result['score']:.4f}, ID: {result['doc_id']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}", exc_info=True)
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.embedding_model.get_text_embedding(text)
            return embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {e}", exc_info=True)
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector index."""
        try:
            index = self.ingestion_service.get_index()
            
            stats = {
                "document_count": len(index.docstore.docs),
                "embedding_model": "text-embedding-3-large",
                "vector_store_type": "SimpleVectorStore"
            }
            
            return stats
        except Exception as e:
            logger.error(f"Error getting index stats: {e}", exc_info=True)
            raise
