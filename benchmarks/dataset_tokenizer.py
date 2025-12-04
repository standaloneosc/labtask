"""Tokenization utilities for datasets."""

from typing import Optional

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class DatasetTokenizer:
    """Helper class for tokenizing dataset prompts."""
    
    def __init__(self, tokenizer_name: Optional[str] = None, tokenizer=None):
        """
        Initialize tokenizer.
        
        Args:
            tokenizer_name: HuggingFace tokenizer name/path
            tokenizer: Pre-initialized tokenizer instance
        """
        self.tokenizer = None
        
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif tokenizer_name and TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    trust_remote_code=True,
                )
            except Exception as e:
                print(f"Warning: Could not load tokenizer {tokenizer_name}: {e}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            except Exception:
                pass
        
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return len(text) // 4
    
    def tokenize(self, text: str) -> list[int]:
        """
        Tokenize text to token IDs.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        if self.tokenizer:
            try:
                return self.tokenizer.encode(text, add_special_tokens=False)
            except Exception:
                pass
        
        # Fallback: return empty list
        return []

