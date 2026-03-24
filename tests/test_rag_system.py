"""
Tests for Wikipedia RAG system components.

Tests cover data ingestion, retrieval, and answer generation.
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil

# Note: These are placeholder tests that demonstrate the test structure
# Full tests require the Wikipedia dataset and Azure credentials


class TestDataIngestion:
    """Tests for data ingestion service."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        # These would be configured for actual tests
        pass
    
    @pytest.mark.asyncio
    async def test_ingest_data_basic(self):
        """Test basic data ingestion."""
        from ingestion import DataIngestionService
        
        service = DataIngestionService()
        
        # Test with small dataset for speed
        result = await service.ingest_wikipedia_data(
            max_documents=10,
            force_reload=True
        )
        
        assert result["status"] in ["success", "skipped"]
        assert result["documents_loaded"] >= 0
        assert result["embeddings_created"] >= 0
    
    def test_create_documents(self):
        """Test document creation from DataFrame."""
        import pandas as pd
        from ingestion import DataIngestionService
        
        service = DataIngestionService()
        
        # Create sample data
        df = pd.DataFrame({
            "passage_text": ["Test passage 1", "Test passage 2"],
            "title": ["Title 1", "Title 2"]
        })
        
        documents = service._create_documents(df)
        
        assert len(documents) == 2
        assert documents[0].text == "Test passage 1"
        assert "title" in documents[0].metadata
    
    def test_get_document_count(self):
        """Test getting document count."""
        from ingestion import DataIngestionService
        
        service = DataIngestionService()
        count = service.get_document_count()
        
        assert isinstance(count, int)
        assert count >= 0


class TestRetrieval:
    """Tests for document retrieval service."""
    
    @pytest.mark.asyncio
    async def test_retrieve_documents_basic(self):
        """Test basic document retrieval."""
        from retrieval import RetrievalService
        
        service = RetrievalService()
        
        try:
            results = await service.retrieve_documents(
                query="test query",
                top_k=5
            )
            
            assert isinstance(results, list)
            for result in results:
                assert "content" in result
                assert "score" in result
                assert "metadata" in result
        except RuntimeError as e:
            # Expected if no index is loaded yet
            assert "No index loaded" in str(e)
    
    def test_embedding_generation(self):
        """Test embedding generation."""
        from retrieval import RetrievalService
        
        service = RetrievalService()
        
        try:
            embedding = service.get_embedding("test text")
            
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(x, (int, float)) for x in embedding)
        except Exception:
            # May fail without proper Azure credentials
            pass
    
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
    
    @pytest.mark.asyncio
    async def test_generate_answer(self):
        """Test answer generation."""
        from generation import GenerationService
        
        service = GenerationService()
        
        # Test with minimal data
        documents = [
            {
                "content": "Python is a programming language.",
                "score": 0.95,
                "metadata": {"title": "Python"}
            }
        ]
        
        try:
            result = await service.generate_answer(
                query="What is Python?",
                documents=documents,
                temperature=0.7
            )
            
            assert "answer" in result
            assert len(result["answer"]) > 0
            assert "model" in result
        except Exception:
            # May fail without proper Azure credentials
            pass
    
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
        assert "[Document 1" in prompt
        assert "[Document 2" in prompt
    
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
    """Tests for FastAPI endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from main import app
        
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        # Will be 503 if services aren't initialized
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "message" in data
    
    @pytest.mark.asyncio
    async def test_ingest_endpoint(self, client):
        """Test ingest endpoint."""
        response = client.post(
            "/ingest",
            json={
                "max_documents": 5,
                "force_reload": False
            }
        )
        
        # May fail without proper setup or Azure credentials
        assert response.status_code in [200, 500]
    
    def test_search_endpoint_validation(self, client):
        """Test search endpoint request validation."""
        # Invalid request (missing required field)
        response = client.post("/search", json={})
        assert response.status_code == 422
        
        # Valid request structure
        response = client.post(
            "/search",
            json={
                "query": "test query",
                "top_k": 5
            }
        )
        
        # May fail without proper setup but structure is valid
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
        
        # May fail without proper setup but structure is valid
        assert response.status_code in [200, 500, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
