import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import random
from utils.byte_tokenizer import ByteTokenizer


class ByteDataset(Dataset):
    """
    Dataset for loading and processing byte data files.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        tokenizer: ByteTokenizer,
        file_pattern: str = "*",
        max_length: Optional[int] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: int = 0,
        transform: Optional[Callable] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        seed: int = 42
    ):
        """
        Initialize the byte dataset.
        
        Args:
            data_dir: Directory containing the data files
            tokenizer: ByteTokenizer instance for encoding the data
            file_pattern: Glob pattern for finding files (default: "*")
            max_length: Maximum sequence length (defaults to tokenizer.max_length)
            chunk_size: Size of chunks to split files into (default: None, use whole files)
            chunk_overlap: Overlap between consecutive chunks (default: 0)
            transform: Optional function to transform the data
            cache_dir: Optional directory to cache processed data
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length or tokenizer.max_length
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Set random seed
        random.seed(seed)
        
        # Find all files matching the pattern
        self.file_paths = list(self.data_dir.glob(file_pattern))
        
        # Create cache directory if needed
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # If using chunks, create list of (file_path, start_idx, end_idx) tuples
        if self.chunk_size:
            self.chunks = []
            for file_path in self.file_paths:
                file_size = file_path.stat().st_size
                for start_idx in range(0, file_size, self.chunk_size - self.chunk_overlap):
                    end_idx = min(start_idx + self.chunk_size, file_size)
                    if end_idx - start_idx < 10:  # Skip very small chunks
                        continue
                    self.chunks.append((file_path, start_idx, end_idx))
        else:
            self.chunks = [(file_path, 0, file_path.stat().st_size) for file_path in self.file_paths]
    
    def __len__(self) -> int:
        """Return the number of chunks in the dataset."""
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a chunk of data by index.
        
        Args:
            idx: Index of the chunk
            
        Returns:
            Dictionary containing:
                - input_ids: Tensor of token IDs
                - attention_mask: Tensor of attention mask
        """
        file_path, start_idx, end_idx = self.chunks[idx]
        
        # Check if cached
        if self.cache_dir:
            cache_file = self.cache_dir / f"{file_path.stem}_{start_idx}_{end_idx}.pt"
            if cache_file.exists():
                return torch.load(cache_file)
        
        # Read the chunk
        with open(file_path, 'rb') as f:
            f.seek(start_idx)
            data = f.read(end_idx - start_idx)
        
        # Apply transform if provided
        if self.transform:
            data = self.transform(data)
        
        # Tokenize the data
        encoding = self.tokenizer.encode_batch(
            [data],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert to single tensors
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
        
        # Cache if needed
        if self.cache_dir:
            torch.save(result, cache_file)
        
        return result


def create_dataloaders(
    data_dir: Union[str, Path],
    tokenizer: ByteTokenizer,
    batch_size: int = 16,
    max_length: Optional[int] = None,
    chunk_size: Optional[int] = None,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation data loaders.
    
    Args:
        data_dir: Directory containing the data files
        tokenizer: ByteTokenizer instance for encoding the data
        batch_size: Batch size for the data loaders
        max_length: Maximum sequence length
        chunk_size: Size of chunks to split files into
        val_split: Fraction of data to use for validation
        num_workers: Number of worker processes for data loading
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create the dataset
    dataset = ByteDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        chunk_size=chunk_size,
        cache_dir=Path(data_dir) / "cache",
        seed=seed
    )
    
    # Split into train and validation sets
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.seed(seed)
    random.shuffle(indices)
    
    # Calculate split point
    val_size = int(np.floor(val_split * dataset_size))
    train_indices, val_indices = indices[val_size:], indices[:val_size]
    
    # Create data loaders
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create validation data loader if needed
    val_dataloader = None
    if val_split > 0:
        val_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(val_indices),
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_dataloader, val_dataloader