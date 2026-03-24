"""
Utility functions for RAG system notebooks.

Provides helpers for data display, metrics calculation, and visualization.
"""

import numpy as np
from typing import List, Dict, Any
from numpy.linalg import norm


def format_relevance_score(score: float) -> str:
    """Format relevance score as a visual bar."""
    percentage = int(score * 100)
    bar_length = percentage // 5
    bar = '█' * bar_length + '░' * (20 - bar_length)
    return f"[{bar}] {score:.4f}"


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (norm(a) * norm(b))


def print_section(title: str, width: int = 70):
    """Print a formatted section header."""
    line = "=" * width
    print(f"\n{line}")
    print(f"{title.center(width)}")
    print(f"{line}\n")


def print_subsection(title: str, width: int = 70):
    """Print a formatted subsection header."""
    line = "-" * width
    print(f"\n{line}")
    print(f"{title}")
    print(f"{line}")


def display_document(doc: Dict[str, Any], index: int = 1, max_chars: int = 200):
    """Format and display a document nicely."""
    print(f"\nDocument {index}:")
    if 'score' in doc:
        print(f"  Relevance: {format_relevance_score(doc['score'])}")
    
    content = doc.get('content', '')
    if len(content) > max_chars:
        content = content[:max_chars] + "..."
    print(f"  Content: {content}")
    
    if doc.get('metadata'):
        print(f"  Metadata: {doc['metadata']}")


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate common statistics for a list of values."""
    if not values:
        return {}
    
    import statistics
    return {
        'count': len(values),
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'stdev': statistics.stdev(values) if len(values) > 1 else 0,
        'min': min(values),
        'max': max(values),
    }


def print_statistics(stats: Dict[str, float], name: str = "Statistics"):
    """Print statistics in a formatted way."""
    print(f"\n{name}:")
    for key, value in stats.items():
        if key == 'count':
            print(f"  {key}: {int(value)}")
        elif key == 'stdev':
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value:.4f}")


def quality_score(answer: str, sources: List[Dict], threshold: float = 0.6) -> Dict[str, Any]:
    """Evaluate answer quality based on multiple factors."""
    metrics = {
        'answer_length': len(answer),
        'answer_words': len(answer.split()),
        'has_citations': '[Document' in answer,
        'source_count': len(sources),
        'avg_source_score': sum(s.get('score', 0) for s in sources) / len(sources) if sources else 0,
    }
    
    # Quality rating
    quality_rating = 0
    if metrics['answer_length'] > 300:
        quality_rating += 1
    if metrics['has_citations']:
        quality_rating += 1
    if metrics['avg_source_score'] > threshold:
        quality_rating += 1
    if metrics['source_count'] >= 3:
        quality_rating += 1
    
    metrics['quality_rating'] = f"{quality_rating}/4"
    
    return metrics


def progress_bar(step: int, total: int, width: int = 50):
    """Display a progress bar."""
    filled = int(width * step / total)
    bar = '█' * filled + '░' * (width - filled)
    percent = 100 * step / total
    print(f"\r[{bar}] {percent:.1f}% ({step}/{total})", end='', flush=True)
    if step == total:
        print()  # New line when done


def format_answer(answer: str, max_lines: int = 10) -> str:
    """Format answer for display with line limitation."""
    lines = answer.split('\n')
    if len(lines) > max_lines:
        return '\n'.join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
    return answer


def embedding_stats(embedding: List[float]) -> Dict[str, Any]:
    """Calculate statistics about an embedding vector."""
    arr = np.array(embedding)
    return {
        'dimension': len(embedding),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'norm': float(np.linalg.norm(arr)),
    }


def similarity_matrix(embeddings: List[List[float]]) -> np.ndarray:
    """Calculate similarity matrix between multiple embeddings."""
    n = len(embeddings)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            else:
                matrix[i, j] = cosine_similarity(embeddings[i], embeddings[j])
    
    return matrix


def print_similarity_matrix(matrix: np.ndarray, labels: List[str] = None):
    """Pretty-print a similarity matrix."""
    n = matrix.shape[0]
    if labels is None:
        labels = [f"Item {i+1}" for i in range(n)]
    
    # Print header
    print("\nSimilarity Matrix:")
    print("  " + " ".join(f"{i:>8}" for i in range(n)))
    
    # Print rows
    for i, label in enumerate(labels):
        row_str = " ".join(f"{matrix[i, j]:8.4f}" for j in range(n))
        print(f"{label:>3} {row_str}")
