"""
Quick startup verification script for the RAG system.

This script tests that all components can be imported and initialized.
"""

import sys
import traceback
import asyncio

def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    
    try:
        from src.ingestion import DataIngestionService
        print("[OK] ingestion module imported")
    except Exception as e:
        print(f"[FAIL] Failed to import ingestion: {e}")
        traceback.print_exc()
        return False
    
    try:
        from src.retrieval import RetrievalService
        print("[OK] retrieval module imported")
    except Exception as e:
        print(f"[FAIL] Failed to import retrieval: {e}")
        traceback.print_exc()
        return False
    
    try:
        from src.generation import GenerationService
        print("[OK] generation module imported")
    except Exception as e:
        print(f"[FAIL] Failed to import generation: {e}")
        traceback.print_exc()
        return False
    
    try:
        from src.main import app
        print("[OK] FastAPI app imported")
    except Exception as e:
        print(f"[FAIL] Failed to import FastAPI app: {e}")
        traceback.print_exc()
        return False
    
    return True


async def test_services():
    """Test that services can be initialized."""
    print("\nTesting service initialization...")
    
    try:
        from src.ingestion import DataIngestionService
        service = DataIngestionService()
        print(f"[OK] DataIngestionService initialized ({service.get_document_count()} documents)")
    except Exception as e:
        print(f"[FAIL] Failed to initialize DataIngestionService: {e}")
        traceback.print_exc()
        return False
    
    try:
        from src.retrieval import RetrievalService
        service = RetrievalService()
        print(f"[OK] RetrievalService initialized")
    except Exception as e:
        print(f"[FAIL] Failed to initialize RetrievalService: {e}")
        traceback.print_exc()
        return False
    
    try:
        from src.generation import GenerationService
        service = GenerationService()
        print(f"[OK] GenerationService initialized")
    except Exception as e:
        print(f"[FAIL] Failed to initialize GenerationService: {e}")
        traceback.print_exc()
        return False
    
    return True


async def main():
    """Run all startup tests."""
    print("=" * 60)
    print("RAG System Startup Verification")
    print("=" * 60)
    
    if not test_imports():
        print("\n[RESULT] Import tests failed!")
        return False
    
    if not await test_services():
        print("\n[RESULT] Service initialization tests failed!")
        return False
    
    print("\n[RESULT] All startup tests passed!")
    print("\nNext steps:")
    print("1. Authenticate with Azure: azd auth login --scope api://ailab/Model.Access")
    print("2. Start the server: uv run python -m src.main")
    print("3. Access API docs: http://localhost:8000/docs")
    print("4. Ingest data: POST /ingest with {\"max_documents\": 100}")
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
