from typing import List, Dict, Union, Optional, Any
import numpy as np
import torch
from pathlib import Path


class ByteTokenizer:
    """
    A tokenizer that works directly with bytes, treating each byte as a token.
    
    This tokenizer is simple but effective for processing raw binary data.
    Each byte value (0-255) is mapped to a unique token ID.
    """
    
    def __init__(self, pad_token_id: int = 0, max_length: int = 2048):
        """
        Initialize the byte tokenizer.
        
        Args:
            pad_token_id: Token ID used for padding (default: 0)
            max_length: Maximum sequence length (default: 2048)
        """
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        
        # Create the vocabulary mapping (byte value to token ID)
        # We reserve token ID 0 for padding
        self.byte_to_id = {i: i + 1 for i in range(255)}
        self.byte_to_id[255] = 0  # Use 255 as padding to keep the mapping simple
        
        # Create the reverse mapping (token ID to byte value)
        self.id_to_byte = {v: k for k, v in self.byte_to_id.items()}
        self.id_to_byte[0] = 255  # Padding token maps to byte 255
        
        # Set vocabulary size (256 bytes + padding token, which reuses one byte value)
        self.vocab_size = 256
    
    def encode(self, raw_bytes: Union[bytes, List[int]], max_length: Optional[int] = None) -> List[int]:
        """
        Encode raw bytes into token IDs.
        
        Args:
            raw_bytes: Raw bytes or list of byte values to encode
            max_length: Optional maximum length (defaults to self.max_length)
            
        Returns:
            List of token IDs
        """
        if max_length is None:
            max_length = self.max_length
        
        # Convert to list of integers if needed
        if isinstance(raw_bytes, bytes):
            byte_values = list(raw_bytes)
        else:
            byte_values = raw_bytes
        
        # Truncate if too long
        if len(byte_values) > max_length:
            byte_values = byte_values[:max_length]
        
        # Convert each byte to its token ID
        token_ids = [self.byte_to_id.get(b, 1) for b in byte_values]  # Default to 1 for unknown bytes
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> bytes:
        """
        Decode token IDs back to raw bytes.
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            Raw bytes
        """
        # Convert each token ID back to its byte value
        byte_values = [self.id_to_byte.get(tid, 0) for tid in token_ids if tid != self.pad_token_id]
        
        # Convert list of integers to bytes
        return bytes(byte_values)
    
    def encode_batch(
        self, 
        batch_raw_bytes: List[Union[bytes, List[int]]], 
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Encode a batch of raw bytes into token IDs.
        
        Args:
            batch_raw_bytes: List of raw bytes or byte lists to encode
            padding: Whether to pad sequences to the maximum length
            truncation: Whether to truncate sequences longer than max_length
            max_length: Optional maximum length (defaults to self.max_length)
            return_tensors: Optional format for the output tensors ('pt' for PyTorch)
            
        Returns:
            Dictionary containing:
                - input_ids: List or tensor of token IDs
                - attention_mask: List or tensor of attention masks
        """
        if max_length is None:
            max_length = self.max_length
        
        # Encode each item in the batch
        batch_token_ids = [self.encode(raw_bytes, max_length if truncation else None) for raw_bytes in batch_raw_bytes]
        
        # Find the maximum length in the batch
        if padding:
            batch_max_length = max(len(ids) for ids in batch_token_ids)
            batch_max_length = min(batch_max_length, max_length)
        else:
            batch_max_length = max_length
        
        # Initialize the attention masks
        attention_masks = []
        
        # Pad the sequences if required
        if padding:
            padded_batch_token_ids = []
            for token_ids in batch_token_ids:
                # Create attention mask (1 for real tokens, 0 for padding)
                mask = [1] * len(token_ids) + [0] * (batch_max_length - len(token_ids))
                attention_masks.append(mask[:batch_max_length])
                
                # Pad the token IDs
                padded_ids = token_ids + [self.pad_token_id] * (batch_max_length - len(token_ids))
                padded_batch_token_ids.append(padded_ids[:batch_max_length])
            
            batch_token_ids = padded_batch_token_ids
        else:
            # If no padding, attention mask is all ones
            attention_masks = [[1] * len(ids) for ids in batch_token_ids]
        
        # Convert to tensors if required
        if return_tensors == 'pt':
            batch_token_ids = torch.tensor(batch_token_ids)
            attention_masks = torch.tensor(attention_masks)
        
        return {
            'input_ids': batch_token_ids,
            'attention_mask': attention_masks
        }
    
    def save_tokenizer(self, path: Union[str, Path]):
        """
        Save the tokenizer configuration to a file.
        
        Args:
            path: Path to save the tokenizer configuration
        """
        import json
        
        # Create the configuration dictionary
        config = {
            'pad_token_id': self.pad_token_id,
            'max_length': self.max_length,
            'vocab_size': self.vocab_size
        }
        
        # Save the configuration
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'ByteTokenizer':
        """
        Load a tokenizer from a configuration file.
        
        Args:
            path: Path to the tokenizer configuration
            
        Returns:
            Initialized ByteTokenizer
        """
        import json
        
        # Load the configuration
        with open(path, 'r') as f:
            config = json.load(f)
        
        # Create the tokenizer
        return cls(
            pad_token_id=config.get('pad_token_id', 0),
            max_length=config.get('max_length', 2048)
        )