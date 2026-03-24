"""
Data ingestion module for Wikipedia RAG system.

Handles loading Wikipedia passages from HuggingFace dataset, creating embeddings,
and populating the vector database.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
import json
from datetime import datetime
import time

import pandas as pd
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.kvstore import SimpleKVStore
from llama_index.core.vector_stores import SimpleVectorStore

try:
    from .llamaindex_models import get_embedding_model
except ImportError:
    from llamaindex_models import get_embedding_model

logger = logging.getLogger(__name__)


class DataIngestionService:
    """Service for ingesting data into the vector database."""
    
    # Configuration
    DATASET_PASSAGES_URL = "hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet"
    DATASET_QUESTIONS_URL = "hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet/part.0.parquet"
    
    PERSIST_DIR = Path("./data/vector_store")
    INDEX_FILE = PERSIST_DIR / "index.json"
    METADATA_FILE = PERSIST_DIR / "metadata.json"
    
    def __init__(self):
        """Initialize the data ingestion service."""
        self.persist_dir = self.PERSIST_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.index = None
        self.embedding_model = None
        self.documents = None
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create a new one."""
        try:
            # Try to load from disk if it exists
            if (self.persist_dir / "docstore.json").exists():
                logger.info("Loading existing vector index...")
                from llama_index.core.storage import StorageContext
                
                # Create storage components
                docstore = SimpleDocumentStore.from_persist_dir(str(self.persist_dir))
                index_store = SimpleIndexStore.from_persist_dir(str(self.persist_dir))
                vector_store = SimpleVectorStore.from_persist_dir(str(self.persist_dir))
                
                storage_context = StorageContext.from_defaults(
                    docstore=docstore,
                    index_store=index_store,
                    vector_store=vector_store,
                )
                
                # Load embedding model for consistency
                self.embedding_model = get_embedding_model()
                self.index = load_index_from_storage(storage_context)
                logger.info("Successfully loaded existing index")
            else:
                logger.info("No existing index found. Will create new index on ingestion.")
                self.index = None
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}. Creating new index.")
            self.index = None
    
    async def ingest_wikipedia_data(
        self,
        max_documents: Optional[int] = None,
        force_reload: bool = False
    ) -> Dict[str, Any]:
        """
        Ingest Wikipedia passages and create vector index.
        
        Args:
            max_documents: Maximum number of documents to load (for testing)
            force_reload: Force reload data even if index exists
            
        Returns:
            Dictionary with ingestion results
        """
        start_time = time.time()
        
        # Check if we should skip loading
        if self.index is not None and not force_reload:
            logger.info("Index already exists and force_reload=False, skipping ingestion")
            return {
                "documents_loaded": len(self.index.docstore.docs),
                "embeddings_created": len(self.index.docstore.docs),
                "status": "skipped"
            }
        
        try:
            logger.info("Starting Wikipedia data ingestion...")
            
            # Step 1: Load passages from HuggingFace
            logger.info(f"Loading passages from {self.DATASET_PASSAGES_URL}")
            passages_df = pd.read_parquet(self.DATASET_PASSAGES_URL)
            logger.info(f"Loaded {len(passages_df)} passages")
            
            # Limit documents if specified (for testing)
            if max_documents:
                passages_df = passages_df.head(max_documents)
                logger.info(f"Limiting to {max_documents} documents for testing")
            
            # Step 2: Convert to LlamaIndex Documents
            logger.info("Converting passages to LlamaIndex documents...")
            documents = self._create_documents(passages_df)
            logger.info(f"Created {len(documents)} documents")
            
            # Step 3: Create embeddings and build index
            logger.info("Creating embeddings and building vector index...")
            self.embedding_model = get_embedding_model()
            
            # Create storage context with individual components
            from llama_index.core.storage import StorageContext
            storage_context = StorageContext.from_defaults(
                docstore=SimpleDocumentStore(),
                index_store=SimpleIndexStore(),
                vector_store=SimpleVectorStore(),
            )
            
            # Build index with embedding model
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=self.embedding_model,
                show_progress=True
            )
            
            # Save index to disk
            storage_context.persist(str(self.persist_dir))
            logger.info(f"Index persisted to {self.persist_dir}")
            
            # Save metadata
            self._save_metadata(len(documents), passages_df)
            
            elapsed = time.time() - start_time
            logger.info(f"Data ingestion completed in {elapsed:.2f} seconds")
            
            return {
                "documents_loaded": len(documents),
                "embeddings_created": len(documents),
                "status": "success",
                "elapsed_seconds": elapsed
            }
            
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}", exc_info=True)
            raise
    
    def _create_documents(self, passages_df: pd.DataFrame) -> List[Document]:
        """
        Convert DataFrame rows to LlamaIndex Documents.
        
        Args:
            passages_df: DataFrame with passage data
            
        Returns:
            List of LlamaIndex Document objects
        """
        documents = []
        
        for idx, row in passages_df.iterrows():
            # Extract passage content and metadata
            passage_text = row.get("passage_text", str(row))
            
            # Create metadata from other columns
            metadata = {}
            for col in passages_df.columns:
                if col != "passage_text" and pd.notna(row[col]):
                    metadata[col] = str(row[col])
            
            # Create LlamaIndex Document
            doc = Document(
                text=passage_text,
                metadata=metadata,
                doc_id=f"doc_{idx}"
            )
            documents.append(doc)
        
        return documents
    
    def _save_metadata(self, doc_count: int, passages_df: pd.DataFrame):
        """Save ingestion metadata."""
        metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "documents_loaded": doc_count,
            "dataset_url": self.DATASET_PASSAGES_URL,
            "embedding_model": "text-embedding-3-large",
            "vector_db_type": "LlamaIndex SimpleVectorStore",
            "dataset_info": {
                "total_rows": len(passages_df),
                "columns": list(passages_df.columns)
            }
        }
        
        with open(self.METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {self.METADATA_FILE}")
    
    def get_index(self):
        """Get the current vector index."""
        if self.index is None:
            raise RuntimeError("No index loaded. Please call ingest_wikipedia_data() first.")
        return self.index
    
    def get_document_count(self) -> int:
        """Get the number of documents in the index."""
        if self.index is None:
            return 0
        return len(self.index.docstore.docs)
    
    def get_metadata(self) -> Optional[Dict]:
        """Load metadata from disk if available."""
        if self.METADATA_FILE.exists():
            with open(self.METADATA_FILE, "r") as f:
                return json.load(f)
        return None
