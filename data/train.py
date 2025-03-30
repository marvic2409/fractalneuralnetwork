import os
import argparse
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union, Any
import numpy as np
from tqdm import tqdm
import logging
import sys
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fractal_model.mamba_model import MambaConfig, FractalMambaModel
from utils.byte_tokenizer import ByteTokenizer
from utils.data_loader import create_dataloaders


def setup_logging(log_dir: Union[str, Path], name: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to save log files
        name: Optional name for the logger
        
    Returns:
        Configured logger
    """
    # Create the log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{name}_{timestamp}" if name else timestamp
    log_file = os.path.join(log_dir, f"{log_name}.log")
    
    # Configure the logger
    logger = logging.getLogger(name or __name__)
    logger.setLevel(logging.INFO)
    
    # Create a file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: Union[str, Path],
    name: Optional[str] = None
) -> str:
    """
    Save a model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        loss: Current loss
        checkpoint_dir: Directory to save checkpoints
        name: Optional name for the checkpoint
        
    Returns:
        Path to the saved checkpoint
    """
    # Create the checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create the checkpoint name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"{name}_epoch{epoch}_{timestamp}.pt" if name else f"checkpoint_epoch{epoch}_{timestamp}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    # Save the checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Tuple[nn.Module, Optional[optim.Optimizer], int, float]:
    """
    Load a model checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load the checkpoint on
        
    Returns:
        Tuple of (model, optimizer, epoch, loss)
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Return model, optimizer, epoch, and loss
    return model, optimizer, checkpoint['epoch'], checkpoint['loss']


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad_norm: Optional[float] = None
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Data loader for training
        optimizer: Optimizer for updating weights
        criterion: Loss function
        device: Device to train on
        clip_grad_norm: Optional gradient clipping norm
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    # Progress bar
    pbar = tqdm(dataloader, desc="Training")
    
    for batch in pbar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Create shifted targets for language modeling
        # Use the input shifted one position to the right as the target
        targets = input_ids.clone()
        targets[:, :-1] = input_ids[:, 1:]
        targets[:, -1] = 0  # Pad the last position
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Calculate loss
        # If output is a dictionary, get the right tensor
        if isinstance(outputs, dict):
            logits = outputs["last_hidden_state"]
        else:
            logits = outputs
        
        # Reshape for loss calculation
        batch_size, seq_len, hidden_size = logits.shape
        
        # Apply a linear layer for token prediction
        # (We create this on the fly to avoid modifying the model architecture)
        prediction_layer = nn.Linear(hidden_size, 256).to(device)  # 256 for byte values
        token_logits = prediction_layer(logits)
        
        # Calculate loss for all non-padding tokens
        loss = criterion(token_logits.view(-1, 256), targets.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients if specified
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        # Update weights
        optimizer.step()
        
        # Update total loss
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix(loss=loss.item())
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Validate the model.
    
    Args:
        model: Model to validate
        dataloader: Data loader for validation
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Average loss for the validation set
    """
    model.eval()
    total_loss = 0.0
    
    # No gradient calculation during validation
    with torch.no_grad():
        # Progress bar
        pbar = tqdm(dataloader, desc="Validation")
        
        for batch in pbar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Create shifted targets for language modeling
            targets = input_ids.clone()
            targets[:, :-1] = input_ids[:, 1:]
            targets[:, -1] = 0  # Pad the last position
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate loss
            if isinstance(outputs, dict):
                logits = outputs["last_hidden_state"]
            else:
                logits = outputs
            
            # Reshape for loss calculation
            batch_size, seq_len, hidden_size = logits.shape
            
            # Apply a linear layer for token prediction
            prediction_layer = nn.Linear(hidden_size, 256).to(device)
            token_logits = prediction_layer(logits)
            
            # Calculate loss for all non-padding tokens
            loss = criterion(token_logits.view(-1, 256), targets.view(-1))
            
            # Update total loss
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix(loss=loss.item())
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss


def train(
    config: Dict[str, Any],
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    checkpoint_dir: Union[str, Path] = 'checkpoints',
    log_dir: Union[str, Path] = 'logs',
    device: Optional[torch.device] = None,
    checkpoint_freq: int = 1,
    resume_from: Optional[str] = None
) -> nn.Module:
    """
    Train the model.
    
    Args:
        config: Training configuration
        model: Model to train
        train_dataloader: Data loader for training
        val_dataloader: Optional data loader for validation
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory to save logs
        device: Device to train on
        checkpoint_freq: Frequency of checkpoints in epochs
        resume_from: Optional checkpoint to resume from
        
    Returns:
        Trained model
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set up logging
    logger = setup_logging(log_dir, name=config.get('name', 'fractal_mamba'))
    logger.info(f"Using device: {device}")
    logger.info(f"Training configuration: {config}")
    
    # Move model to device
    model = model.to(device)
    
    # Create optimizer
    optimizer_name = config.get('optimizer', 'adam').lower()
    learning_rate = config.get('learning_rate', 1e-4)
    weight_decay = config.get('weight_decay', 0.01)
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Create learning rate scheduler
    scheduler_name = config.get('scheduler', 'cosine').lower()
    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.get('epochs', 10),
            eta_min=config.get('min_learning_rate', 1e-6)
        )
    elif scheduler_name == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('scheduler_factor', 0.5),
            patience=config.get('scheduler_patience', 2)
        )
    elif scheduler_name == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    # Create loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        model, optimizer, start_epoch, _ = load_checkpoint(resume_from, model, optimizer, device)
        start_epoch += 1  # Start from the next epoch
    
    # Training loop
    num_epochs = config.get('epochs', 10)
    clip_grad_norm = config.get('clip_grad_norm', None)
    
    logger.info(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        start_time = time.time()
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device, clip_grad_norm)
        epoch_time = time.time() - start_time
        
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, train loss: {train_loss:.4f}")
        
        # Validate if a validation dataloader is provided
        val_loss = float('inf')
        if val_dataloader:
            val_loss = validate(model, val_dataloader, criterion, device)
            logger.info(f"Validation loss: {val_loss:.4f}")
        
        # Update scheduler if using one that depends on validation loss
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save checkpoint if it's time
        if (epoch + 1) % checkpoint_freq == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = save_checkpoint(
                model, optimizer, epoch + 1, val_loss if val_dataloader else train_loss,
                checkpoint_dir, config.get('name', 'fractal_mamba')
            )
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model if using validation
        if val_dataloader and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = save_checkpoint(
                model, optimizer, epoch + 1, val_loss,
                checkpoint_dir, config.get('name', 'fractal_mamba_best')
            )
            logger.info(f"New best model saved to {best_checkpoint_path}")
    
    logger.info("Training completed")
    
    return model


def main():
    """Main function for training."""
    parser = argparse.ArgumentParser(description="Train a Fractal-Mamba model")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create model configuration
    model_config = MambaConfig(
        hidden_size=config.get('hidden_size', 768),
        intermediate_size=config.get('intermediate_size', 1536),
        num_hidden_layers=config.get('num_hidden_layers', 8),
        state_size=config.get('state_size', 16),
        time_step_rank=config.get('time_step_rank', 4),
        conv_kernel=config.get('conv_kernel', 4),
        expand_factor=config.get('expand_factor', 2),
        fractal_count_per_layer=config.get('fractal_count', 3),
        fractal_depth_range=tuple(config.get('fractal_depth_range', (2, 5))),
        fractal_dim_range=tuple(config.get('fractal_dim_range', (1.2, 2.4))),
        dropout_prob=config.get('dropout_prob', 0.1),
        vocab_size=256,  # Fixed for byte tokenizer
        max_position_embeddings=config.get('max_length', 2048)
    )
    
    # Create model
    model = FractalMambaModel(model_config)
    
    # Create tokenizer
    tokenizer = ByteTokenizer(
        max_length=config.get('max_length', 2048)
    )
    
    # Create data loaders
    train_dataloader, val_dataloader = create_dataloaders(
        data_dir=config.get('data_dir', 'data'),
        tokenizer=tokenizer,
        batch_size=config.get('batch_size', 16),
        max_length=config.get('max_length', 2048),
        chunk_size=config.get('chunk_size', None),
        val_split=config.get('val_split', 0.1),
        num_workers=config.get('num_workers', 4),
        seed=config.get('seed', 42)
    )
    
    # Set device
    device_name = config.get('device', 'auto')
    if device_name == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)
    
    # Train model
    train(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        checkpoint_dir=config.get('checkpoint_dir', 'checkpoints'),
        log_dir=config.get('log_dir', 'logs'),
        device=device,
        checkpoint_freq=config.get('checkpoint_freq', 1),
        resume_from=args.resume
    )


if __name__ == "__main__":
    main()