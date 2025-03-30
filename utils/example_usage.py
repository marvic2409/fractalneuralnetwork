


import sys
import os
from pathlib import Path
import torch
import numpy as np
import argparse
import time
import logging

from fractal_model.mamba_model import MambaConfig, FractalMambaModel
from utils.byte_tokenizer import ByteTokenizer
from utils.device_utils import optimize_device_settings, summarize_model



def example_inference(
    model: FractalMambaModel,
    input_file: Path,
    max_length: int = 1024,
    device_name: str = 'auto',
    logger: logging.Logger = None
) -> torch.Tensor:
    """
    Run inference with a Fractal-Mamba model on example data.
    
    Args:
        model: Model to use for inference
        input_file: Input file to process
        max_length: Maximum sequence length
        device_name: Device to use
        logger: Logger for logging information
        
    Returns:
        Model output tensor
    """
    # Set device
    device, _ = optimize_device_settings(device_name, logger)
    
    # Create tokenizer
    tokenizer = ByteTokenizer(max_length=max_length)
    
    # Read input file
    if logger:
        logger.info(f"Reading input file: {input_file}")
    
    with open(input_file, 'rb') as f:
        data = f.read()
    
    # Tokenize input
    if logger:
        logger.info(f"Tokenizing input (max_length={max_length})")
    
    encoding = tokenizer.encode_batch(
        [data[:max_length]],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Run inference
    if logger:
        logger.info("Running inference")
    
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        inference_time = time.time() - start_time
    
    if logger:
        logger.info(f"Inference completed in {inference_time:.2f}s")
        
        # Log output shape
        if isinstance(outputs, dict):
            output_tensor = outputs['last_hidden_state']
        else:
            output_tensor = outputs
        
        logger.info(f"Output shape: {output_tensor.shape}")
    
    return outputs


def main():
    """Main function for example usage."""
    parser = argparse.ArgumentParser(description="Example usage of Fractal-Mamba model")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size for the model")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--fractal_count", type=int, default=2, help="Number of fractals per layer")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--create_sample", action="store_true", help="Create sample data")
    parser.add_argument("--sample_size_mb", type=int, default=1, help="Size of sample data in MB")
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    # Create sample data if requested
    if args.create_sample:
        sample_path = Path("data/sample.bin")
        logger.info(f"Creating sample data of size {args.sample_size_mb} MB at {sample_path}")
        create_sample_data(sample_path, size_mb=args.sample_size_mb)
    
    # Create model
    logger.info("Creating Fractal-Mamba model")
    model = example_create_model(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        fractal_count=args.fractal_count,
        device_name=args.device,
        logger=logger
    )
    
    # Run inference with sample data
    sample_path = Path("data/sample.bin")
    if sample_path.exists():
        logger.info("Running example inference")
        example_inference(
            model=model,
            input_file=sample_path,
            max_length=args.max_length,
            device_name=args.device,
            logger=logger
        )
    else:
        logger.info(f"Sample data not found at {sample_path}. Use --create_sample to create it.")
    
    logger.info("Example completed")


if __name__ == "__main__":
    main()

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))


def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger


def create_sample_data(output_path: Path, size_mb: int = 1) -> None:
    """
    Create sample data file for testing.
    
    Args:
        output_path: Path to save the sample data
        size_mb: Size of the sample data in MB
    """
    # Create directory if it doesn't exist
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Generate random bytes
    num_bytes = size_mb * 1024 * 1024
    data = np.random.bytes(num_bytes)
    
    # Save to file
    with open(output_path, 'wb') as f:
        f.write(data)


def example_create_model(
    hidden_size: int = 256,
    num_layers: int = 4,
    fractal_count: int = 2,
    device_name: str = 'auto',
    logger: logging.Logger = None
) -> FractalMambaModel:
    """
    Create a Fractal-Mamba model with example settings.
    
    Args:
        hidden_size: Hidden size for the model
        num_layers: Number of layers
        fractal_count: Number of fractals per layer
        device_name: Device to use
        logger: Logger for logging information
        
    Returns:
        Created model
    """
    # Set device
    device, settings = optimize_device_settings(device_name, logger)
    
    if logger:
        logger.info(f"Creating model with hidden_size={hidden_size}, num_layers={num_layers}, fractal_count={fractal_count}")
        logger.info(f"Device optimization settings: {settings}")
    
    # Create model configuration
    config = MambaConfig(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        state_size=16,
        time_step_rank=4,
        conv_kernel=4,
        expand_factor=2,
        fractal_count_per_layer=fractal_count,
        fractal_depth_range=(2, 4),
        fractal_dim_range=(1.2, 2.0),
        dropout_prob=0.1,
        vocab_size=256,  # Fixed for byte tokenizer
        max_position_embeddings=1024
    )
    
    # Create model
    model = FractalMambaModel(config)
    
    # Move model to device
    model = model.to(device)
    
    # Summarize model
    if logger:
        summarize_model(model, logger=logger)
    
    return model