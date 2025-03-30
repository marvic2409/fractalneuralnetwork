import torch
import torch.nn as nn
import os
import psutil
from typing import Union, Optional, Tuple, Dict, Any, List
import logging
import numpy as np


def optimize_device_settings(
    device_name: str = 'auto',
    logger: Optional[logging.Logger] = None
) -> Tuple[torch.device, Dict[str, Any]]:
    """
    Optimize device settings for training or inference.
    
    Args:
        device_name: Device name ('auto', 'cuda', 'cuda:0', 'cpu', etc.)
        logger: Optional logger for logging device information
        
    Returns:
        Tuple of (device, optimization_settings)
    """
    # Initialize settings dictionary
    settings = {}
    
    # Auto-detect device if needed
    if device_name == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            if logger:
                logger.info(f"Auto-detected CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            if logger:
                logger.info("No CUDA device available, using CPU")
    else:
        device = torch.device(device_name)
        if logger:
            if device.type == 'cuda' and device.index is not None:
                logger.info(f"Using CUDA device {device.index}: {torch.cuda.get_device_name(device.index)}")
            elif device.type == 'cuda':
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                logger.info(f"Using device: {device.type}")
    
    # Optimize based on device type
    if device.type == 'cuda':
        # Get CUDA device properties
        props = torch.cuda.get_device_properties(device if device.index is not None else 0)
        
        # Get GPU memory
        total_memory = props.total_memory / (1024 ** 3)  # Convert to GB
        
        if logger:
            logger.info(f"GPU Memory: {total_memory:.2f} GB")
            logger.info(f"CUDA Capability: {props.major}.{props.minor}")
            logger.info(f"Number of SMs: {props.multi_processor_count}")
        
        # Optimize CUDA settings
        settings['cudnn_benchmark'] = True  # Enable cuDNN benchmarking
        torch.backends.cudnn.benchmark = True
        
        # Autotuned suggested batch size based on GPU memory
        if total_memory < 4:
            settings['suggested_batch_size'] = 4
        elif total_memory < 8:
            settings['suggested_batch_size'] = 8
        elif total_memory < 16:
            settings['suggested_batch_size'] = 16
        else:
            settings['suggested_batch_size'] = 32
        
        # Optimize based on CUDA capability
        if props.major >= 7:
            # Enable TF32 precision (Ampere and later)
            if props.major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                settings['allow_tf32'] = True
            
            # Enable mixed precision
            settings['use_amp'] = True
            settings['grad_scaler'] = torch.cuda.amp.GradScaler()
    else:
        # CPU optimizations
        # Get number of CPU cores
        cpu_cores = psutil.cpu_count(logical=False)
        cpu_threads = psutil.cpu_count(logical=True)
        
        # Get available RAM
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        
        if logger:
            logger.info(f"CPU physical cores: {cpu_cores}")
            logger.info(f"CPU logical cores: {cpu_threads}")
            logger.info(f"RAM: {ram_gb:.2f} GB")
        
        # Optimize number of workers for data loading
        settings['suggested_num_workers'] = max(1, cpu_cores - 1)
        
        # Optimize batch size based on available RAM
        if ram_gb < 8:
            settings['suggested_batch_size'] = 4
        elif ram_gb < 16:
            settings['suggested_batch_size'] = 8
        else:
            settings['suggested_batch_size'] = 16
        
        # Optimize PyTorch CPU settings if available
        if hasattr(torch, 'set_num_threads'):
            num_threads = max(1, cpu_threads - 2)  # Leave some threads for the system
            torch.set_num_threads(num_threads)
            settings['num_threads'] = num_threads
        
        # Disable AMP for CPU
        settings['use_amp'] = False
    
    return device, settings


def enable_mixed_precision(model: nn.Module) -> Tuple[nn.Module, bool]:
    """
    Enable mixed precision for the model if supported.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (model, is_using_amp)
    """
    if torch.cuda.is_available():
        # Create AMP context manager
        use_amp = True
        
        # Apply mixed precision to model
        model = model.to(memory_format=torch.channels_last)
        
        return model, use_amp
    else:
        return model, False


def get_model_size(model: nn.Module) -> float:
    """
    Calculate the model size in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    
    # Calculate size in MB (assuming float32 parameters)
    size_mb = params * 4 / (1024 ** 2)
    
    return size_mb


def get_gpu_memory_usage() -> Dict[int, float]:
    """
    Get GPU memory usage for all available GPUs.
    
    Returns:
        Dictionary mapping GPU indices to memory usage in GB
    """
    if torch.cuda.is_available():
        memory_usage = {}
        
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # Convert to GB
            memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)  # Convert to GB
            
            memory_usage[i] = {
                'allocated': memory_allocated,
                'reserved': memory_reserved
            }
        
        return memory_usage
    else:
        return {}


def distribute_model(model: nn.Module, devices: List[torch.device]) -> nn.Module:
    """
    Distribute model across multiple devices if available.
    
    Args:
        model: PyTorch model
        devices: List of devices to distribute across
        
    Returns:
        Distributed model
    """
    if len(devices) <= 1:
        # Only one device, no distribution needed
        return model.to(devices[0])
    
    # Check if all devices are CUDA
    if all(device.type == 'cuda' for device in devices):
        # Use DataParallel for multiple GPUs
        gpu_ids = [device.index if device.index is not None else 0 for device in devices]
        model = nn.DataParallel(model, device_ids=gpu_ids)
        return model.to(devices[0])  # Move model to first device
    else:
        # Mixed device types not supported
        return model.to(devices[0])


def summarize_model(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 256),
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Summarize model architecture and parameters.
    
    Args:
        model: PyTorch model
        input_size: Input size for the model summary
        logger: Optional logger for logging the summary
    """
    def log_or_print(message: str) -> None:
        if logger:
            logger.info(message)
        else:
            print(message)
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size
    model_size_mb = get_model_size(model)
    
    # Print summary
    log_or_print("=" * 80)
    log_or_print(f"Model Summary: {model.__class__.__name__}")
    log_or_print("=" * 80)
    log_or_print(f"Total Parameters: {total_params:,}")
    log_or_print(f"Trainable Parameters: {trainable_params:,}")
    log_or_print(f"Model Size: {model_size_mb:.2f} MB")
    log_or_print("=" * 80)
    
    # Print layer information if available
    if hasattr(model, 'named_parameters'):
        log_or_print("\nLayer Information:")
        log_or_print("-" * 80)
        log_or_print(f"{'Name':<50} {'Shape':<20} {'Parameters':<12}")
        log_or_print("-" * 80)
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                log_or_print(f"{name:<50} {str(list(param.shape)):<20} {param.numel():,}")
        
        log_or_print("=" * 80)