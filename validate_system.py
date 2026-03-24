#!/usr/bin/env python
"""
Integrated validation script for Wikipedia RAG system.

Automates Phase 1 validation:
1. Start FastAPI server
2. Test /health endpoint
3. Ingest Wikipedia data (small dataset)
4. Test /search endpoint
5. Test /query (full RAG) endpoint
6. Report results with statistics

Usage:
    uv run python validate_system.py
"""

import asyncio
import subprocess
import sys
import time
import requests
import json
from pathlib import Path
from typing import Dict, Any, Optional


class RAGValidator:
    """Validates the complete RAG system."""
    
    def __init__(self, host: str = "http://localhost:8000", timeout: int = 120):
        """Initialize validator.
        
        Args:
            host: FastAPI server address
            timeout: Timeout for various operations (seconds)
        """
        self.host = host
        self.timeout = timeout
        self.server_process = None
        self.results = {
            "health_check": None,
            "data_ingestion": None,
            "search": None,
            "query": None,
            "errors": []
        }
    
    def print_header(self, title: str, width: int = 70):
        """Print formatted header."""
        line = "=" * width
        print(f"\n{line}")
        print(f"{title.center(width)}")
        print(f"{line}\n")
    
    def print_step(self, step: int, title: str):
        """Print formatted step."""
        print(f"\n[STEP {step}] {title}")
        print("-" * 70)
    
    def print_success(self, msg: str):
        """Print success message."""
        print(f"[OK] {msg}")
    
    def print_error(self, msg: str):
        """Print error message."""
        print(f"[FAIL] {msg}")
    
    def start_server(self) -> bool:
        """Start FastAPI server in background."""
        self.print_step(1, "Start FastAPI Server")
        
        try:
            # Start server in background
            self.server_process = subprocess.Popen(
                ["uv", "run", "python", "-m", "src.main"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print("Starting FastAPI server...")
            
            # Wait for server to be ready
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                try:
                    response = requests.get(f"{self.host}/health", timeout=2)
                    if response.status_code == 200:
                        self.print_success(f"Server ready at {self.host}")
                        self.results["health_check"] = True
                        return True
                except requests.ConnectionError:
                    time.sleep(1)
            
            self.print_error(f"Server failed to start within {self.timeout}s")
            self.results["health_check"] = False
            return False
        
        except Exception as e:
            self.print_error(f"Failed to start server: {e}")
            self.results["health_check"] = False
            self.results["errors"].append(str(e))
            return False
    
    def test_health(self) -> bool:
        """Test /health endpoint."""
        self.print_step(2, "Test Health Endpoint")
        
        try:
            response = requests.get(f"{self.host}/health", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                self.print_success(f"Health check passed: {data['status']}")
                self.print_success(f"Message: {data['message']}")
                return True
            else:
                self.print_error(f"Health check failed with status {response.status_code}")
                return False
        
        except Exception as e:
            self.print_error(f"Health check error: {e}")
            self.results["errors"].append(f"Health check: {str(e)}")
            return False
    
    def test_ingestion(self, max_docs: int = 50) -> bool:
        """Test /ingest endpoint."""
        self.print_step(3, "Test Data Ingestion")
        
        try:
            payload = {
                "max_documents": max_docs,
                "force_reload": True
            }
            
            print(f"Ingesting {max_docs} Wikipedia documents...")
            print("(This may take a few minutes...)")
            
            response = requests.post(
                f"{self.host}/ingest",
                json=payload,
                timeout=600  # Long timeout for data loading
            )
            
            if response.status_code == 200:
                data = response.json()
                self.results["data_ingestion"] = data
                
                self.print_success(f"Status: {data['status']}")
                self.print_success(f"Documents loaded: {data['documents_loaded']}")
                self.print_success(f"Embeddings created: {data['embeddings_created']}")
                self.print_success(f"Message: {data['message']}")
                
                return data['documents_loaded'] > 0
            else:
                self.print_error(f"Ingestion failed with status {response.status_code}")
                self.print_error(f"Response: {response.text}")
                return False
        
        except Exception as e:
            self.print_error(f"Ingestion error: {e}")
            self.results["errors"].append(f"Ingestion: {str(e)}")
            return False
    
    def test_search(self) -> bool:
        """Test /search endpoint."""
        self.print_step(4, "Test Search Endpoint")
        
        test_queries = [
            "What is artificial intelligence?",
            "machine learning algorithms",
            "neural networks"
        ]
        
        try:
            for query in test_queries:
                payload = {
                    "query": query,
                    "top_k": 3
                }
                
                print(f"\nQuery: {query}")
                response = requests.post(
                    f"{self.host}/search",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self.print_success(f"Retrieved {data['count']} documents")
                    
                    for i, match in enumerate(data['matches'][:2], 1):
                        score = match['score']
                        content_preview = match['content'][:100]
                        print(f"  Document {i} (score: {score:.4f}): {content_preview}...")
                else:
                    self.print_error(f"Search failed: {response.status_code}")
                    return False
            
            self.results["search"] = True
            return True
        
        except Exception as e:
            self.print_error(f"Search error: {e}")
            self.results["errors"].append(f"Search: {str(e)}")
            return False
    
    def test_rag_query(self) -> bool:
        """Test /query endpoint (full RAG pipeline)."""
        self.print_step(5, "Test RAG Query Endpoint")
        
        test_questions = [
            "What is artificial intelligence?",
            "How does machine learning work?",
        ]
        
        try:
            for question in test_questions:
                payload = {
                    "query": question,
                    "top_k": 3,
                    "temperature": 0.7
                }
                
                print(f"\nQuestion: {question}")
                response = requests.post(
                    f"{self.host}/query",
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self.print_success(f"Answer generated successfully")
                    
                    # Display answer preview
                    answer_preview = data['answer'][:200]
                    print(f"Answer: {answer_preview}...")
                    
                    # Display sources
                    source_count = len(data['sources'])
                    self.print_success(f"Sources cited: {source_count} documents")
                    
                else:
                    self.print_error(f"Query failed: {response.status_code}")
                    print(f"Response: {response.text}")
                    return False
            
            self.results["query"] = True
            return True
        
        except Exception as e:
            self.print_error(f"Query error: {e}")
            self.results["errors"].append(f"Query: {str(e)}")
            return False
    
    def print_summary(self):
        """Print validation summary."""
        self.print_header("VALIDATION SUMMARY")
        
        checks = [
            ("Health Check", self.results["health_check"]),
            ("Data Ingestion", self.results["data_ingestion"] is not None),
            ("Search Endpoint", self.results["search"]),
            ("RAG Query Endpoint", self.results["query"]),
        ]
        
        passed = 0
        for check_name, passed_check in checks:
            if passed_check:
                print(f"[OK] {check_name:<30} PASSED")
                passed += 1
            else:
                print(f"[FAIL] {check_name:<30} FAILED")
        
        print(f"\nResult: {passed}/{len(checks)} checks passed")
        
        # Show ingestion stats if available
        if self.results["data_ingestion"]:
            ing = self.results["data_ingestion"]
            print(f"\nData Ingestion Stats:")
            print(f"  Documents loaded: {ing['documents_loaded']}")
            print(f"  Embeddings created: {ing['embeddings_created']}")
        
        # Show errors if any
        if self.results["errors"]:
            print(f"\nErrors encountered:")
            for error in self.results["errors"]:
                print(f"  - {error}")
        
        print("\n" + "=" * 70)
        
        if passed == len(checks):
            print("[OK] ALL CHECKS PASSED - System is operational!".center(70))
            print("\nNext steps:")
            print("  1. Open Jupyter Lab: uv run jupyter lab notebooks/")
            print("  2. Run notebook 01_data_inspection.ipynb to explore loaded data")
            print("  3. Run notebook 03_rag_pipeline.ipynb for more examples")
        else:
            print("[FAIL] SOME CHECKS FAILED - Review errors above".center(70))
        
        print("=" * 70)
    
    def cleanup(self):
        """Cleanup: stop server."""
        if self.server_process:
            print("\n[CLEANUP] Stopping FastAPI server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
                print("[OK] Server stopped")
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                print("[OK] Server force-stopped")
    
    async def run_validation(self):
        """Run complete validation suite."""
        self.print_header("WIKIPEDIA RAG SYSTEM - PHASE 1 VALIDATION")
        
        try:
            # Step 1: Start server
            if not self.start_server():
                return
            
            # Wait a bit for full initialization
            await asyncio.sleep(2)
            
            # Step 2: Health check
            if not self.test_health():
                return
            
            # Step 3: Data ingestion
            if not self.test_ingestion(max_docs=50):
                return
            
            # Step 4: Search
            if not self.test_search():
                return
            
            # Step 5: RAG query
            if not self.test_rag_query():
                return
            
        finally:
            self.print_summary()
            self.cleanup()


async def main():
    """Main entry point."""
    validator = RAGValidator()
    await validator.run_validation()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nValidation failed: {e}")
        sys.exit(1)
