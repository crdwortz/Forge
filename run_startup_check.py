#!/usr/bin/env python
"""
Launcher script to properly initialize the RAG system.
Sets up PYTHONPATH and imports all modules.
"""
import sys
from pathlib import Path

# Add src to path so packages can be imported correctly
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    print("RAG System - Module Import Test")
    print("-" * 50)
    
    try:
        from llamaindex_models import get_embedding_model, get_chat_model
        print("[OK] llamaindex_models")
    except Exception as e:
        print(f"[FAIL] llamaindex_models: {e}")
        sys.exit(1)
    
    try:
        from ingestion import DataIngestionService
        print("[OK] ingestion")
    except Exception as e:
        print(f"[FAIL] ingestion: {e}")
        sys.exit(1)
    
    try:
        from retrieval import RetrievalService
        print("[OK] retrieval")
    except Exception as e:
        print(f"[FAIL] retrieval: {e}")
        sys.exit(1)
    
    try:
        from generation import GenerationService
        print("[OK] generation")
    except Exception as e:
        print(f"[FAIL] generation: {e}")
        sys.exit(1)
    
    try:
        from main import app
        print("[OK] main (FastAPI app)")
    except Exception as e:
        print(f"[FAIL] main: {e}")
        sys.exit(1)
    
    print("-" * 50)
    print("All modules imported successfully!")
    print("\nTo start the server:")
    print("  uv run python -m src.main")
    print("\nFor API documentation:")
    print("  http://localhost:8000/docs")
