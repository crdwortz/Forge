# Setup
This is the README for the RAG system you are building. For instructions to the consultant leading the project, see the [readme_setup](./readme_setup.md).

# Wikipedia RAG System

A Retrieval-Augmented Generation (RAG) system using LlamaIndex with Wikipedia articles, built with FastAPI and Azure OpenAI models.

## Overview

This project implements a complete RAG pipeline that:
- Loads Wikipedia passages from HuggingFace datasets
- Generates embeddings using Azure OpenAI's `text-embedding-3-large`
- Stores vectors in a local LlamaIndex vector database
- Retrieves relevant documents based on semantic similarity
- Generates answers using Azure OpenAI's `gpt-4o` model

## Features

- 🔒 **Secure Model Access**: All Azure OpenAI models accessed through controlled authentication
- 🚀 **FastAPI Backend**: RESTful API for all RAG operations
- 📊 **Full Observability**: Jupyter notebooks for inspecting every stage of the pipeline
- 🎯 **Explainable**: Clear separation of retrieval and generation steps
- 🧪 **Evaluation Ready**: Built-in support for test questions and metrics

## Architecture

```
User Query → Embedding → Vector Search → Retrieve Top-K Docs → Augment Prompt → GPT-4o → Answer
```

See `docs/architecture.md` for detailed architecture diagrams.

## Prerequisites

1. **Python 3.13+**
2. **Azure Access**: Access to Azure OpenAI services with AI Lab models
3. **Azure CLI**: For authentication
4. **UV Package Manager**: Recommended for dependency management

## Installation

### 1. Clone and Setup

```bash
cd /path/to/your_project

# Set up virtual environment and install dependencies using uv
uv venv
uv sync
```

### 2. Authenticate with Azure

```bash
# Login with AI Lab scope
azd auth login --scope api://ailab/Model.Access
```

This authentication is required for accessing the Azure OpenAI models (GPT-4o and text-embedding-3-large).

See `docs/authentication.md` for more details on authentication setup.

## Usage

### Starting the API Server

```bash
# Start the FastAPI server
uv run python -m src.main

# Server will be available at http://localhost:8000
```

### Web Interface (Easiest Way)

For a user-friendly chat interface, open `frontend.html` in your browser:

```bash
# Start the server first
uv run python -m src.main

# Then open the frontend in your browser
# Double-click frontend.html or open it with your browser
```

The web interface provides:
- **Chat-style interface** for asking questions
- **Example questions** to get started
- **Real-time responses** from the RAG system
- **Source information** showing how many documents were used

### API Documentation

Once the server is running, access the interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Ingest Data
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"max_documents": 100, "force_reload": false}'
```

#### Search Documents
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "top_k": 5}'
```

#### Generate Answer (Full RAG Pipeline)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is artificial intelligence?", "top_k": 5, "temperature": 0.7}'
```

## Observability with Jupyter Notebooks

The system includes comprehensive Jupyter notebooks for exploring and understanding each stage of the RAG pipeline.

### Running Notebooks

```bash
# Start Jupyter Lab
uv run jupyter lab notebooks/

# Or start Jupyter Notebook
uv run jupyter notebook notebooks/
```

### Available Notebooks

#### 1. **01_data_inspection.ipynb** - Data & Embeddings Analysis
Explore the loaded Wikipedia data and vector database:
- Load and initialize the vector index
- Examine document statistics
- Analyze embedding vectors
- Inspect metadata
- View sample documents from the database

**Use this to**: Understand what data is stored and how embeddings look

#### 2. **02_retrieval_demo.ipynb** - Semantic Search & Document Retrieval
Demonstrate the document retrieval pipeline:
- Execute semantic searches with various queries
- Compare relevance scores across queries
- Analyze embedding similarity
- Examine top-k search results
- Understand relevance scoring

**Use this to**: See how the system finds relevant documents

#### 3. **03_rag_pipeline.ipynb** - End-to-End RAG Demonstration
Show the complete question-answering pipeline:
- Walk through each step: query → embedding → search → augment → generate
- Generate answers to multiple test questions
- Analyze answer quality and metrics
- Explore temperature effects on responses
- Understand the full workflow

**Use this to**: See the complete system in action

### Notebook Helper Utilities

Import utilities for common notebook operations:

```python
from notebook_utils import (
    format_relevance_score,
    cosine_similarity,
    print_section,
    display_document,
    calculate_statistics,
    quality_score,
)
```

Available helpers:
- `format_relevance_score()` - Visual score display
- `cosine_similarity()` - Calculate vector similarity
- `print_section()` - Formatted output headers
- `calculate_statistics()` - Summary statistics
- `quality_score()` - Answer quality evaluation
- `embedding_stats()` - Embedding vector analysis


## Project Structure

```
rag-project/
├── src/                           # Main application code
│   ├── main.py                   # FastAPI application & 4 REST endpoints
│   ├── ingestion.py              # DataIngestionService - load & embed data
│   ├── retrieval.py              # RetrievalService - semantic search
│   ├── generation.py             # GenerationService - LLM answer generation
│   ├── llamaindex_models.py      # Model isolation layer
│   ├── ailab/                    # AI Lab authentication utilities
│   │   └── utils/
│   │       └── azure.py
│   └── __init__.py
│
├── notebooks/                     # Jupyter notebooks for observability
│   ├── 01_data_inspection.ipynb   # Explore data & embeddings
│   ├── 02_retrieval_demo.ipynb    # Semantic search demonstration
│   ├── 03_rag_pipeline.ipynb      # End-to-end RAG examples
│   └── notebook_utils.py          # Helper utilities for notebooks
│
├── tests/                         # Test suite
│   └── test_rag_system.py        # 13 tests for all components
│
├── data/                          # Data directory (created at runtime)
│   └── vector_store/             # LlamaIndex vector database storage
│
├── docs/                          # Documentation
│   ├── architecture.md           # System architecture diagrams
│   ├── authentication.md         # Azure authentication setup
│   ├── model_isolation.md        # Model access patterns
│   ├── testing.md                # Testing guide
│   └── examples.md               # Usage examples
│
├── pyproject.toml                # Project dependencies & config
├── README.md                     # This file
├── STATUS.md                     # Implementation checklist
├── instructions.md               # Project instructions
├── frontend.html                 # Web interface for asking questions
└── verify_startup.py             # Startup verification script
```

### Core Modules

- **main.py**: FastAPI application with 4 REST endpoints (health, ingest, search, query)
- **ingestion.py**: Loads Wikipedia data, creates embeddings, manages vector DB
- **retrieval.py**: Handles semantic search and document retrieval
- **generation.py**: Augments prompts and calls LLM for answer generation
- **llamaindex_models.py**: Secure, isolated access to Azure OpenAI models
## Testing

### Run Tests

```bash
# Run all tests with verbose output
uv run pytest tests/ -v

# Run specific test class
uv run pytest tests/test_rag_system.py::TestRetrieval -v

# Run with coverage
uv run pytest tests/ --cov=src
```

### Test Suite

The project includes 13 comprehensive tests:
- **DataIngestion Tests** (3): Document creation, ingestion, metadata
- **Retrieval Tests** (3): Document retrieval, embedding generation, index stats
- **Generation Tests** (4): Answer generation, prompt augmentation, quality validation
- **FastAPI Tests** (3): Endpoint validation and request/response models

### Verification

Quick startup verification:
```bash
uv run python run_startup_check.py
```

This verifies all modules import and initialize successfully.

## Data Sources

The system uses the `rag-mini-wikipedia` dataset from HuggingFace:
- **Passages**: `hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet`
- **Test Questions**: `hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet/part.0.parquet`

## Model Configuration

This project uses controlled access to Azure OpenAI models:

- **Embedding Model**: `text-embedding-3-large` (2024-10-01-preview)
- **Chat Model**: `gpt-4o` (2024-10-01-preview)

All model access goes through the `llamaindex_models.py` isolation layer. See `docs/model_isolation.md` for details.

## Development Workflow

### 1. First Time Setup
```bash
# Authenticate
azd auth login --scope api://ailab/Model.Access

# Install dependencies
uv venv
uv sync

# Start API server

```

### 2. Ingest Data


### 3. Explore with Notebooks


### 4. Iterate and Observe


## Troubleshooting

### Authentication Issues
```bash
# Verify Azure login
azd auth login --scope api://ailab/Model.Access

# Check token
azd auth token --output json | jq -r '.expiresOn'
azd auth token --output json | jq -r '.token'

# remove token
azd auth logout
```

### Import Errors
```bash
# Reinstall dependencies
uv sync --reinstall
```

### Index Not Found
If you get "No index available" errors:
```bash
# Run ingestion first
```

### Dataset Loading Issues

## Testing

The system includes comprehensive integration tests that verify both internal behavior and API consistency.

### Quick Start

See `docs/testing.md` for detailed testing guide.

### Test Organization

- **Internal Tests** 
- **Integration Tests**  
- **Logs**:  

## Troubleshooting

This project follows two core principles:

1. **Simple (a la Rich Hickey)**: Independent, unentangled components in a clear data pipeline
2. **Explainable (a la "Rewilding Software Engineering")**: Observable, inspectable steps from query to answer

See `instructions.md` for the complete project brief.

## Examples

See the `docs/llamaindex_examples/example_*.py` files for standalone demonstrations:
- `example_model_isolation.py` - Model access patterns
- `example_chat_usage.py` - LLM completions
- `example_vector_search.py` - Vector similarity search

## Documentation

Comprehensive documentation is available in the `docs/` directory:
- **Architecture**: System design and data flow
- **Authentication**: Azure setup and model access
- **Model Isolation**: Security and controlled access patterns
- **Examples**: Usage patterns and best practices


