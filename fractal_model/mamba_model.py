import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

try:
    from causal_conv1d import SSMLinearKernel, SelectiveScanKernel
except ImportError:
    raise ImportError(
        "Please install causal-conv1d package with: pip install causal-conv1d"
    )

from fractal_model.fractal_module import FractalLayer


@dataclass
class MambaConfig:
    """Configuration for the Mamba model."""
    hidden_size: int = 768
    intermediate_size: int = 1536
    num_hidden_layers: int = 8
    num_attention_heads: Optional[int] = None  # Not used in Mamba but kept for compatibility
    state_size: int = 16
    time_step_rank: int = 4
    conv_kernel: int = 4
    expand_factor: int = 2
    fractal_count_per_layer: int = 3
    fractal_depth_range: Tuple[int, int] = (2, 5)
    fractal_dim_range: Tuple[float, float] = (1.2, 2.4)
    dropout_prob: float = 0.1
    vocab_size: int = 256  # For byte tokenizer
    max_position_embeddings: int = 2048
    pad_token_id: int = 0
    layer_norm_epsilon: float = 1e-5


class MambaBlock(nn.Module):
    """
    A Mamba model block integrating Selective State Space models (SSM) with fractal processing.
    
    Based on the Mamba paper: https://arxiv.org/abs/2312.00752
    """
    
    def __init__(self, config: MambaConfig, layer_idx: int):
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.expand_factor = config.expand_factor
        self.expanded_size = self.expand_factor * self.hidden_size
        
        # Layer normalization
        self.input_layernorm = nn.LayerNorm(
            self.hidden_size, eps=config.layer_norm_epsilon
        )
        
        # Projection to higher dimension for SSM
        self.in_proj = nn.Linear(
            self.hidden_size, self.expanded_size * 2, bias=True
        )
        
        # 1D Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.expanded_size,
            out_channels=self.expanded_size,
            kernel_size=config.conv_kernel,
            padding=config.conv_kernel - 1,
            groups=self.expanded_size,
            bias=True,
        )
        
        # Create the selective scan (SSM) kernel
        self.ssm_kernel = SSMLinearKernel(
            self.expanded_size,
            config.state_size,
            config.time_step_rank,
            activation="swish",
        )
        
        # Mixer for different frequency features
        self.mixer = SelectiveScanKernel(
            self.expanded_size,
            config.state_size,
            config.time_step_rank,
            activation="swish",
        )
        
        # Output projection
        self.out_proj = nn.Linear(
            self.expanded_size, self.hidden_size, bias=True
        )
        
        # Fractal processing layer
        self.fractal_layer = FractalLayer(
            dim=self.hidden_size,
            fractal_count=config.fractal_count_per_layer,
            depth_range=config.fractal_depth_range,
            dim_range=config.fractal_dim_range,
            dropout_prob=config.dropout_prob
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the Mamba block.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Mask tensor of shape [batch_size, seq_len]
            output_hidden_states: Whether to return hidden states in addition to output
            
        Returns:
            Output tensor or tuple of output and hidden states
        """
        # Store input for residual connection
        residual = hidden_states
        
        # Apply layer normalization
        hidden_states = self.input_layernorm(hidden_states)
        
        # Apply input projection to get expanded dimensions and gates
        # Shape: [batch_size, seq_len, expanded_size * 2]
        proj_states = self.in_proj(hidden_states)
        
        # Split into x and gate components
        x, gate = torch.split(proj_states, self.expanded_size, dim=-1)
        
        # Apply 1D convolution for local context
        # Need to transpose to [batch_size, channels, seq_len] for conv1d
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)
        
        # Remove extra padding from causal convolution
        x_conv = x_conv[:, :, :x.size(1)]
        
        # Transpose back to [batch_size, seq_len, expanded_size]
        x = x_conv.transpose(1, 2)
        
        # Apply Selective Scan for long-range dependencies
        if attention_mask is not None:
            # Convert attention mask to float for compatibility
            scan_mask = attention_mask.to(x.dtype)
            x = self.mixer(x, mask=scan_mask)
        else:
            x = self.mixer(x)
        
        # Apply silu gate
        x = torch.sigmoid(gate) * x
        
        # Project back to hidden dimension
        hidden_states = self.out_proj(x)
        
        # Apply residual connection
        hidden_states = hidden_states + residual
        
        # Apply fractal processing
        hidden_states = self.fractal_layer(hidden_states)
        
        if output_hidden_states:
            return hidden_states, hidden_states
        
        return hidden_states


class FractalMambaModel(nn.Module):
    """
    Complete Fractal-Mamba neural network model that processes byte sequences
    using recursive probabilistic fractals and Mamba architecture.
    """
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        
        # Embedding layer for byte tokens
        self.embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        
        # Embedding dropout
        self.embedding_dropout = nn.Dropout(config.dropout_prob)
        
        # Stack of Mamba blocks
        self.layers = nn.ModuleList(
            [MambaBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        
        # Final layer normalization
        self.final_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Use Kaiming normal for linear layers
            torch.nn.init.kaiming_normal_(
                module.weight, a=math.sqrt(5), mode="fan_in", nonlinearity="leaky_relu"
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # Use normal distribution for embeddings
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # Use ones and zeros for layer norm
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the Fractal-Mamba model.
        
        Args:
            input_ids: Tensor of shape [batch_size, seq_len] with token IDs
            attention_mask: Tensor of shape [batch_size, seq_len] with attention mask
            position_ids: Tensor of shape [batch_size, seq_len] with position IDs
            output_hidden_states: Whether to return all hidden states
            
        Returns:
            Output tensor or dictionary with outputs and hidden states
        """
        batch_size, seq_length = input_ids.shape[:2]
        
        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Generate attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, seq_length, device=input_ids.device
            )
        
        # Get token embeddings
        hidden_states = self.embeddings(input_ids)
        
        # Add position embeddings
        position_embeddings = self.position_embeddings(position_ids)
        hidden_states = hidden_states + position_embeddings
        
        # Apply embedding dropout
        hidden_states = self.embedding_dropout(hidden_states)
        
        # Store all hidden states if requested
        all_hidden_states = [] if output_hidden_states else None
        
        # Process through each layer
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
        
        # Apply final layer normalization
        hidden_states = self.final_layernorm(hidden_states)
        
        # Gather outputs
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
            return {
                "last_hidden_state": hidden_states,
                "hidden_states": all_hidden_states,
            }
        
        return hidden_states


class FractalMambaForSequenceClassification(nn.Module):
    """Fractal-Mamba model for sequence classification."""
    
    def __init__(self, config: MambaConfig, num_labels: int = 2):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        
        # Core Fractal-Mamba model
        self.mamba = FractalMambaModel(config)
        
        # Classification head
        self.classifier = nn.Linear(config.hidden_size, num_labels)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for sequence classification."""
        outputs = self.mamba(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=False,
        )
        
        # Use the hidden state of the last token for classification
        if isinstance(outputs, dict):
            last_hidden_state = outputs["last_hidden_state"]
        else:
            last_hidden_state = outputs
        
        # Use mean pooling for sequence classification
        pooled_output = torch.mean(last_hidden_state, dim=1)
        
        # Apply the classification head
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": last_hidden_state,
        }


class FractalMambaForRegression(nn.Module):
    """Fractal-Mamba model for regression tasks."""
    
    def __init__(self, config: MambaConfig, num_outputs: int = 1):
        super().__init__()
        self.config = config
        self.num_outputs = num_outputs
        
        # Core Fractal-Mamba model
        self.mamba = FractalMambaModel(config)
        
        # Regression head
        self.regressor = nn.Linear(config.hidden_size, num_outputs)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for regression."""
        outputs = self.mamba(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=False,
        )
        
        # Use the entire sequence for regression
        if isinstance(outputs, dict):
            last_hidden_state = outputs["last_hidden_state"]
        else:
            last_hidden_state = outputs
        
        # Use mean pooling
        pooled_output = torch.mean(last_hidden_state, dim=1)
        
        # Apply the regression head
        predictions = self.regressor(pooled_output)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(predictions.view(-1), targets.view(-1))
        
        return {
            "loss": loss,
            "predictions": predictions,
            "hidden_states": last_hidden_state,
        }