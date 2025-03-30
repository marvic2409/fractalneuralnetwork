import argparse
import torch
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import numpy as np
import yaml
import time
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fractal_model.mamba_model import MambaConfig, FractalMambaModel
from utils.byte_tokenizer import ByteTokenizer
from utils.data_loader import create_dataloaders
from evaluation.inference import setup_logging, load_model


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    metrics: List[str] = ['loss', 'accuracy'],
    prediction_layer: Optional[torch.nn.Linear] = None
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader for evaluation
        device: Device to evaluate on
        metrics: List of metrics to compute
        prediction_layer: Optional prediction layer for token prediction
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    # Create metrics dictionary
    metrics_dict = {metric: 0.0 for metric in metrics}
    
    # Create prediction layer if needed
    if prediction_layer is None and ('loss' in metrics or 'accuracy' in metrics or 'perplexity' in metrics):
        hidden_size = model.config.hidden_size
        prediction_layer = torch.nn.Linear(hidden_size, 256).to(device)  # 256 for byte values
    
    # Create loss function
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=0)  # Ignore padding tokens
    
    total_tokens = 0
    total_correct = 0
    total_loss = 0.0
    
    # Evaluate
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Create targets (shifted input)
            targets = input_ids.clone()
            targets[:, :-1] = input_ids[:, 1:]
            targets[:, -1] = 0  # Pad the last position
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get logits
            if isinstance(outputs, dict):
                hidden_states = outputs['last_hidden_state']
            else:
                hidden_states = outputs
            
            # Apply prediction layer if computing loss or accuracy
            if prediction_layer is not None:
                logits = prediction_layer(hidden_states)
                
                # Calculate loss if needed
                if 'loss' in metrics or 'perplexity' in metrics:
                    loss = loss_fn(logits.view(-1, 256), targets.view(-1))
                    total_loss += loss.item()
                
                # Calculate accuracy if needed
                if 'accuracy' in metrics:
                    # Get predictions
                    predictions = torch.argmax(logits, dim=-1)
                    
                    # Count correct predictions (ignoring padding)
                    mask = (targets != 0).float()
                    correct = ((predictions == targets).float() * mask).sum().item()
                    tokens = mask.sum().item()
                    
                    total_correct += correct
                    total_tokens += tokens
    
    # Compute final metrics
    if 'loss' in metrics:
        metrics_dict['loss'] = total_loss / total_tokens if total_tokens > 0 else float('inf')
    
    if 'perplexity' in metrics:
        metrics_dict['perplexity'] = torch.exp(torch.tensor(total_loss / total_tokens)).item() if total_tokens > 0 else float('inf')
    
    if 'accuracy' in metrics:
        metrics_dict['accuracy'] = total_correct / total_tokens if total_tokens > 0 else 0.0
    
    return metrics_dict


def visualize_metrics(
    metrics: Dict[str, float],
    output_dir: Union[str, Path],
    model_name: str = 'fractal_mamba'
) -> None:
    """
    Visualize evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Directory to save visualizations
        model_name: Name of the model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics as JSON
    metrics_path = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create bar chart for all metrics
    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values())
    plt.title(f"{model_name} Evaluation Metrics")
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.tight_layout()
    
    # Save the figure
    chart_path = os.path.join(output_dir, f"{model_name}_metrics.png")
    plt.savefig(chart_path)
    plt.close()


def main():
    """Main function for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a Fractal-Mamba model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, help="Path to model configuration")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing evaluation data")
    parser.add_argument("--output_dir", type=str, default="evaluation/results", help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="auto", help="Device to run evaluation on")
    parser.add_argument("--metrics", nargs='+', default=['loss', 'accuracy', 'perplexity'], help="Metrics to compute")
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
    
    # Create data loader
    logger.info(f"Creating data loader for {args.data_dir}")
    _, eval_dataloader = create_dataloaders(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        val_split=1.0,  # Use all data for evaluation
        num_workers=4,
        seed=42
    )
    
    # Evaluate model
    logger.info("Evaluating model")
    start_time = time.time()
    metrics = evaluate_model(
        model=model,
        dataloader=eval_dataloader,
        device=device,
        metrics=args.metrics
    )
    evaluation_time = time.time() - start_time
    
    # Log metrics
    logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Visualize metrics
    logger.info(f"Saving evaluation results to {args.output_dir}")
    visualize_metrics(metrics, args.output_dir)
    
    logger.info("Evaluation completed")


if __name__ == "__main__":
    main()