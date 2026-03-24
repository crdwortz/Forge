"""
Answer generation module for Wikipedia RAG system.

Handles augmenting prompts with retrieved documents and generating answers using the LLM.
"""

import logging
from typing import List, Dict, Any, Optional

try:
    from .llamaindex_models import get_chat_model
except ImportError:
    from llamaindex_models import get_chat_model

logger = logging.getLogger(__name__)


class GenerationService:
    """Service for generating answers using retrieved documents."""
    
    def __init__(self):
        """Initialize the generation service."""
        self.chat_model = get_chat_model()
    
    async def generate_answer(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate an answer to the query using retrieved documents.
        
        Args:
            query: The original user query
            documents: List of retrieved documents with content, score, metadata
            temperature: LLM temperature for response generation (0.0 to 1.0)
            
        Returns:
            Dictionary with answer, tokens_used, etc.
        """
        try:
            logger.info(f"Generating answer for query: {query}")
            logger.info(f"Using {len(documents)} retrieved documents")
            
            # Step 1: Create the augmented prompt
            augmented_prompt = self._create_augmented_prompt(query, documents)
            logger.debug(f"Augmented prompt length: {len(augmented_prompt)} characters")
            
            # Step 2: Call the LLM
            response = await self._call_llm(augmented_prompt, temperature)
            logger.info("Successfully generated answer")
            
            return {
                "answer": response["content"],
                "tokens_used": response.get("tokens_used", {}),
                "model": "gpt-4o"
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}", exc_info=True)
            raise
    
    def _create_augmented_prompt(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> str:
        """
        Create an augmented prompt with context from retrieved documents.
        
        Args:
            query: The original user query
            documents: List of retrieved documents
            
        Returns:
            Augmented prompt string
        """
        # Build context from documents
        context_sections = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            score = doc.get("score", 0.0)
            
            # Add document with relevance score
            context_sections.append(f"[Document {i} - Relevance: {score:.3f}]\n{content}")
        
        context = "\n\n---\n\n".join(context_sections)
        
        # Create the augmented prompt
        prompt = f"""You are a helpful assistant that answers questions based on provided context.

## Context (Retrieved from Wikipedia)
{context}

## Question
{query}

## Instructions
1. Answer the question based on the provided context
2. If the context doesn't contain relevant information, say so clearly
3. Keep your answer concise and well-structured
4. Cite which documents you used (e.g., "According to Document 1...")

## Answer"""
        
        return prompt
    
    async def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Call the Azure OpenAI LLM and get a response.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Temperature for response generation
            
        Returns:
            Dictionary with response content and metadata
        """
        try:
            logger.info("Calling LLM for response generation")
            
            # Create chat completion
            response = self.chat_model.complete(
                prompt,
                temperature=temperature
            )
            
            # Extract response content
            answer = response.message.content if hasattr(response, 'message') else str(response)
            
            # Try to extract token usage
            tokens_used = {}
            if hasattr(response, 'raw') and isinstance(response.raw, dict):
                if 'usage' in response.raw:
                    tokens_used = {
                        "prompt_tokens": response.raw['usage'].get('prompt_tokens', 0),
                        "completion_tokens": response.raw['usage'].get('completion_tokens', 0),
                        "total_tokens": response.raw['usage'].get('total_tokens', 0)
                    }
            
            logger.info(f"LLM response received. Answer length: {len(answer)} characters")
            
            return {
                "content": answer,
                "tokens_used": tokens_used,
                "model": "gpt-4o"
            }
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}", exc_info=True)
            raise
    
    def validate_answer_quality(
        self,
        answer: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate the quality of the generated answer.
        
        Args:
            answer: The generated answer
            documents: The retrieved documents used
            
        Returns:
            Quality metrics
        """
        metrics = {
            "answer_length": len(answer),
            "answer_words": len(answer.split()),
            "has_citations": "[Document" in answer,
            "document_count": len(documents),
            "avg_document_score": sum(d.get("score", 0) for d in documents) / len(documents) if documents else 0
        }
        
        return metrics
