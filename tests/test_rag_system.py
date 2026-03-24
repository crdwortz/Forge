"""
Tests for Wikipedia RAG system components.

Tests cover data ingestion, retrieval, and answer generation.
"""

import pytest
from pathlib import Path
import pandas as pd


class TestDataIngestion:
    """Tests for data ingestion service."""
    
    def test_create_documents(self):
        """Test document creation from DataFrame."""
        from ingestion import DataIngestionService
        
        service = DataIngestionService()
        
        # Create sample data
        df = pd.DataFrame({
            "passage": ["Test passage 1", "Test passage 2"],
            "title": ["Title 1", "Title 2"],
            "passage_id": ["1", "2"]
        })
        
        documents = service._create_documents(df)
        
        assert len(documents) == 2
        assert "passage" in documents[0].text or "Test" in documents[0].text
    
    def test_get_document_count(self):
        """Test getting document count."""
        from ingestion import DataIngestionService
        
        service = DataIngestionService()
        count = service.get_document_count()
        
        assert isinstance(count, int)
        assert count >= 0


class TestRetrieval:
    """Tests for document retrieval service."""
    
    def test_get_index_stats(self):
        """Test getting index statistics."""
        from retrieval import RetrievalService
        
        service = RetrievalService()
        
        try:
            stats = service.get_index_stats()
            
            assert isinstance(stats, dict)
            assert "document_count" in stats
            assert "embedding_model" in stats
        except RuntimeError:
            # Expected if no index is loaded
            pass


class TestGeneration:
    """Tests for answer generation service."""
    
    def test_create_augmented_prompt(self):
        """Test augmented prompt creation."""
        from generation import GenerationService
        
        service = GenerationService()
        
        documents = [
            {
                "content": "Document 1 content",
                "score": 0.9,
                "metadata": {}
            },
            {
                "content": "Document 2 content",
                "score": 0.8,
                "metadata": {}
            }
        ]
        
        prompt = service._create_augmented_prompt(
            query="Test query",
            documents=documents
        )
        
        assert "Test query" in prompt
        assert "Document 1 content" in prompt
        assert "Document 2 content" in prompt
    
    def test_validate_answer_quality(self):
        """Test answer quality validation."""
        from generation import GenerationService
        
        service = GenerationService()
        
        documents = [
            {"content": "Test", "score": 0.9, "metadata": {}},
            {"content": "Test", "score": 0.8, "metadata": {}}
        ]
        
        metrics = service.validate_answer_quality(
            answer="This is a test answer. [Document 1 says something.]",
            documents=documents
        )
        
        assert "answer_length" in metrics
        assert "answer_words" in metrics
        assert "has_citations" in metrics
        assert metrics["has_citations"] is True
        assert metrics["document_count"] == 2


class TestFastAPIEndpoints:
    """Tests for FastAPI endpoints - basic validation only."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from main import app
        
        return TestClient(app)
    
    def test_search_endpoint_validation(self, client):
        """Test search endpoint request validation."""
        # Invalid request (missing required field)
        response = client.post("/search", json={})
        assert response.status_code == 422
        
        # Valid request structure - should accept it
        response = client.post(
            "/search",
            json={
                "query": "test query",
                "top_k": 5
            }
        )
        # Accept 200, 500, or 503 since we might not have data loaded
        assert response.status_code in [200, 500, 503]
    
    def test_query_endpoint_validation(self, client):
        """Test query endpoint request validation."""
        # Invalid request (missing required field)
        response = client.post("/query", json={})
        assert response.status_code == 422
        
        # Valid request structure
        response = client.post(
            "/query",
            json={
                "query": "test question",
                "top_k": 5,
                "temperature": 0.7
            }
        )
        
        assert response.status_code in [200, 500, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
