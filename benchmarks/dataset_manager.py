"""Dataset management for benchmarking."""

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class BenchmarkRequest:
    """Represents a benchmark request."""
    prompt: str
    prompt_length: int
    expected_output_length: int
    request_id: str
    metadata: dict = None


class Dataset(ABC):
    """Base class for benchmark datasets."""
    
    @abstractmethod
    def sample_requests(self, num_requests: int, seed: Optional[int] = None) -> List[BenchmarkRequest]:
        """Sample requests from the dataset."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get dataset name."""
        pass


class RandomDataset(Dataset):
    """Random synthetic dataset for controlled benchmarking."""
    
    def __init__(self, 
                 prompt_length: int = 512,
                 output_length: int = 128,
                 vocab_size: int = 50000,
                 seed: int = 42):
        """
        Initialize random dataset.
        
        Args:
            prompt_length: Average prompt length in tokens
            output_length: Average output length in tokens
            vocab_size: Vocabulary size for token generation
            seed: Random seed
        """
        self.prompt_length = prompt_length
        self.output_length = output_length
        self.vocab_size = vocab_size
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def get_name(self) -> str:
        return f"random_p{self.prompt_length}_o{self.output_length}"
    
    def sample_requests(self, num_requests: int, seed: Optional[int] = None) -> List[BenchmarkRequest]:
        """Generate random requests."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = self.rng
        
        requests = []
        for i in range(num_requests):
            # Add some variance to lengths
            prompt_len = max(1, int(rng.normal(self.prompt_length, self.prompt_length * 0.1)))
            output_len = max(1, int(rng.normal(self.output_length, self.output_length * 0.1)))
            
            # Generate random token IDs
            prompt_tokens = rng.randint(0, self.vocab_size, size=prompt_len).tolist()
            prompt_text = f"[TOKENS:{len(prompt_tokens)}]"  # Placeholder
            
            requests.append(BenchmarkRequest(
                prompt=prompt_text,
                prompt_length=prompt_len,
                expected_output_length=output_len,
                request_id=f"random_{i}",
                metadata={"token_ids": prompt_tokens}
            ))
        
        return requests


class ShareGPTDataset(Dataset):
    """ShareGPT conversation dataset."""
    
    def __init__(self, dataset_path: str, seed: int = 42):
        """
        Initialize ShareGPT dataset.
        
        Args:
            dataset_path: Path to ShareGPT JSON file
            seed: Random seed
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._load_data()
    
    def _load_data(self):
        """Load dataset from file."""
        with open(self.dataset_path, 'r') as f:
            self.data = json.load(f)
        
        # Filter and prepare conversations
        self.conversations = []
        for item in self.data:
            if isinstance(item, dict) and 'conversations' in item:
                conversations = item['conversations']
                # Extract user messages (prompts)
                user_messages = [c['value'] for c in conversations 
                               if c.get('from') == 'human']
                if user_messages:
                    self.conversations.append(user_messages[0])
    
    def get_name(self) -> str:
        return f"sharegpt_{self.dataset_path.stem}"
    
    def sample_requests(self, num_requests: int, seed: Optional[int] = None) -> List[BenchmarkRequest]:
        """Sample requests from ShareGPT dataset."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = self.rng
        
        # Estimate token lengths (rough approximation: 1 token ≈ 4 chars)
        requests = []
        available = self.conversations[:]
        
        if len(available) < num_requests:
            # Repeat dataset if needed
            available = available * ((num_requests // len(available)) + 1)
        
        selected = rng.choice(len(available), size=num_requests, replace=False)
        
        for idx, conv_idx in enumerate(selected):
            prompt_text = available[conv_idx]
            # Rough token estimation
            prompt_length = len(prompt_text) // 4
            # Assume average output length
            output_length = prompt_length // 2
            
            requests.append(BenchmarkRequest(
                prompt=prompt_text,
                prompt_length=prompt_length,
                expected_output_length=output_length,
                request_id=f"sharegpt_{idx}",
            ))
        
        return requests


class C4Dataset(Dataset):
    """C4 (Colossal Clean Crawled Corpus) dataset."""
    
    def __init__(self, dataset_path: Optional[str] = None, seed: int = 42):
        """
        Initialize C4 dataset.
        
        Args:
            dataset_path: Optional path to C4 dataset file
            seed: Random seed
        """
        self.dataset_path = dataset_path
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.data = []
        
        if dataset_path:
            self._load_data()
    
    def _load_data(self):
        """Load C4 dataset."""
        # For now, create synthetic C4-like data
        # In production, you'd load from HuggingFace datasets or file
        pass
    
    def get_name(self) -> str:
        return "c4"
    
    def sample_requests(self, num_requests: int, seed: Optional[int] = None) -> List[BenchmarkRequest]:
        """Sample requests from C4 dataset."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = self.rng
        
        # Generate C4-like document excerpts
        requests = []
        for i in range(num_requests):
            # Generate random document-like text
            doc_length = rng.randint(100, 1000)
            prompt_text = f"Document excerpt {i} with length {doc_length}"
            prompt_length = doc_length // 4  # Rough token estimate
            output_length = rng.randint(50, 200)
            
            requests.append(BenchmarkRequest(
                prompt=prompt_text,
                prompt_length=prompt_length,
                expected_output_length=output_length,
                request_id=f"c4_{i}",
            ))
        
        return requests


class DatasetManager:
    """Manages multiple datasets for benchmarking."""
    
    def __init__(self):
        self.datasets: dict[str, Dataset] = {}
    
    def register_dataset(self, name: str, dataset: Dataset):
        """Register a dataset."""
        self.datasets[name] = dataset
    
    def get_dataset(self, name: str) -> Dataset:
        """Get a dataset by name."""
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found. Available: {list(self.datasets.keys())}")
        return self.datasets[name]
    
    def list_datasets(self) -> List[str]:
        """List all registered datasets."""
        return list(self.datasets.keys())
    
    @staticmethod
    def create_default_datasets() -> 'DatasetManager':
        """Create a dataset manager with default datasets."""
        manager = DatasetManager()
        
        # Register default datasets
        manager.register_dataset("random_short", RandomDataset(prompt_length=128, output_length=64))
        manager.register_dataset("random_medium", RandomDataset(prompt_length=512, output_length=128))
        manager.register_dataset("random_long", RandomDataset(prompt_length=2048, output_length=256))
        manager.register_dataset("c4", C4Dataset())
        
        return manager

