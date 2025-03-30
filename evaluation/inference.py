import argparse
import torch
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
import numpy as np
import yaml
import time

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fractal_model.mamba_model import MambaConfig, FractalMambaModel
from utils.byte_tokenizer import ByteTokenizer


def setup_logging(log_file: Optional[Union[str, Path]] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Optional path to log file
        
    Returns:
        Configured logger
    """
    # Configure the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    handlers = []
    
    # Always add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    handlers.append(console_handler)
    
    # Add file handler if log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        handlers.append(file_handler)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add formatter to handlers and handlers to logger
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def load_model(
    model_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None
) -> torch.nn.Module:
    """
    Load a model from a checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        config_path: Optional path to the model configuration
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load configuration if provided
    if config_path:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create model configuration
        config = MambaConfig(
            hidden_size=config_dict.get('hidden_size', 768),
            intermediate_size=config_dict.get('intermediate_size', 1536),
            num_hidden_layers=config_dict.get('num_hidden_layers', 8),
            state_size=config_dict.get('state_size', 16),
            time_step_rank=config_dict.get('time_step_rank', 4),
            conv_kernel=config_dict.get('conv_kernel', 4),
            expand_factor=config_dict.get('expand_factor', 2),
            fractal_count_per_layer=config_dict.get('fractal_count', 3),
            fractal_depth_range=tuple(config_dict.get('fractal_depth_range', (2, 5))),
            fractal_dim_range=tuple(config_dict.get('fractal_dim_range', (1.2, 2.4))),
            dropout_prob=config_dict.get('dropout_prob', 0.1),
            vocab_size=256,  # Fixed for byte tokenizer
            max_position_embeddings=config_dict.get('max_length', 2048)
        )
        
        # Create model
        model = FractalMambaModel(config)
    else:
        # Try to infer the model configuration from the checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'config' in checkpoint:
            # If configuration is stored in the checkpoint
            config_dict = checkpoint['config']
            config = MambaConfig(**config_dict)
            model = FractalMambaModel(config)
        else:
            # Use default configuration and hope it matches
            config = MambaConfig()
            model = FractalMambaModel(config)
    
    # Load the model weights
    model.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict'])
    
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    return model


def process_file(
    model: torch.nn.Module,
    tokenizer: ByteTokenizer,
    input_path: Union[str, Path],
    max_length: Optional[int] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 1
) -> torch.Tensor:
    """
    Process a file through the model.
    
    Args:
        model: Model to use for processing
        tokenizer: Tokenizer for encoding the input
        input_path: Path to the input file
        max_length: Maximum length of the input sequence
        device: Device to process on
        batch_size: Batch size for processing
        
    Returns:
        Model output tensor
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Read the input file
    with open(input_path, 'rb') as f:
        data = f.read()
    
    # Tokenize the input
    if max_length is None:
        max_length = tokenizer.max_length
    
    # Split the data into chunks if necessary
    chunks = []
    for i in range(0, len(data), max_length):
        chunk = data[i:i + max_length]
        chunks.append(chunk)
    
    # Process each chunk in batches
    all_outputs = []
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        
        # Tokenize the batch
        batch_encoding = tokenizer.encode_batch(
            batch_chunks,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move to device
        batch_input_ids = batch_encoding['input_ids'].to(device)
        batch_attention_mask = batch_encoding['attention_mask'].to(device)
        
        # Process through the model
        with torch.no_grad():
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask
            )
        
        # Store the outputs
        if isinstance(outputs, dict):
            outputs = outputs['last_hidden_state']
        
        all_outputs.append(outputs.cpu())
    
    # Concatenate all outputs
    if len(all_outputs) > 1:
        return torch.cat(all_outputs, dim=0)
    else:
        return all_outputs[0]


def main():
    """Main function for inference."""
    parser = argparse.ArgumentParser(description="Run inference with a Fractal-Mamba model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, help="Path to model configuration")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input file")
    parser.add_argument("--output_file", type=str, help="Path to output file")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--device", type=str, default="auto", help="Device to run inference on")
    parser.add_argument("--log_file", type=str, help="Path to log file")
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_file)
    
    # Set device
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, args.config_path, device)
    
    # Create tokenizer
    tokenizer = ByteTokenizer(max_length=args.max_length)
    
    # Process input file
    logger.info(f"Processing input file: {args.input_file}")
    start_time = time.time()
    outputs = process_file(
        model=model,
        tokenizer=tokenizer,
        input_path=args.input_file,
        max_length=args.max_length,
        device=device,
        batch_size=args.batch_size
    )
    processing_time = time.time() - start_time
    logger.info(f"Processing completed in {processing_time:.2f}s")
    
    # Save outputs if specified
    if args.output_file:
        logger.info(f"Saving outputs to {args.output_file}")
        torch.save(outputs, args.output_file)
    
    logger.info("Inference completed")


if __name__ == "__main__":
    main()