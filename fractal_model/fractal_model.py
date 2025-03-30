import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


class RecursiveProbabilisticFractal(nn.Module):
    """
    A module that implements a recursive probabilistic fractal for neural network processing.
    
    The fractal recursively processes input data through a tree-like structure,
    with randomized connection patterns based on probability distributions.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int,
        fractal_dim: float,
        dropout_prob: float = 0.1,
        activation: nn.Module = nn.GELU(),
        seed: Optional[int] = None
    ):
        """
        Initialize the recursive probabilistic fractal.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            depth: Maximum recursion depth of the fractal
            fractal_dim: Fractal dimension parameter (controls branching factor)
            dropout_prob: Dropout probability
            activation: Activation function
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.fractal_dim = fractal_dim
        self.activation = activation
        
        # Determine branching factor based on fractal dimension
        self.branch_factor = max(2, int(self.fractal_dim * 2))
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, output_dim)
        
        # Create the recursive structure
        if depth > 0:
            # Fractal branch weights (probabilistic connections)
            self.branch_weights = nn.Parameter(
                torch.rand(self.branch_factor) + 0.5
            )
            
            # Create the fractal branches
            self.branches = nn.ModuleList([
                self._create_branch(output_dim, i)
                for i in range(self.branch_factor)
            ])
        
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def _create_branch(self, dim: int, branch_idx: int) -> nn.Module:
        """Create a branch of the fractal."""
        # Each branch can have a slightly different structure
        branch_depth = max(1, self.depth - 1)
        branch_dim = int(dim * (0.8 + 0.4 * torch.sigmoid(self.branch_weights[branch_idx]).item()))
        
        if branch_depth > 0 and branch_idx % 2 == 0:
            # Some branches are recursive fractals (with reduced depth)
            return RecursiveProbabilisticFractal(
                input_dim=dim,
                output_dim=dim,
                depth=branch_depth,
                fractal_dim=self.fractal_dim * (0.7 + 0.6 * torch.rand(1).item()),
                dropout_prob=self.dropout.p
            )
        else:
            # Some branches are feed-forward networks
            return nn.Sequential(
                nn.Linear(dim, branch_dim),
                self.activation,
                nn.Linear(branch_dim, dim)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the fractal."""
        # Initial projection
        out = self.input_proj(x)
        
        # If we have branches (depth > 0), process through fractal structure
        if hasattr(self, 'branches'):
            # Get probabilistic weights for branch mixing
            weights = F.softmax(self.branch_weights, dim=0)
            
            # Process through each branch and combine weighted results
            branch_outputs = []
            for i, branch in enumerate(self.branches):
                branch_out = branch(out)
                branch_outputs.append(branch_out * weights[i])
            
            # Sum branch outputs
            branch_sum = torch.stack(branch_outputs).sum(dim=0)
            out = out + branch_sum  # Residual connection
        
        # Apply layer normalization and dropout
        out = self.layer_norm(out)
        out = self.dropout(out)
        
        return out


class FractalLayer(nn.Module):
    """
    A layer composed of multiple fractal modules.
    """
    
    def __init__(
        self,
        dim: int,
        fractal_count: int,
        depth_range: Tuple[int, int],
        dim_range: Tuple[float, float],
        dropout_prob: float = 0.1
    ):
        """
        Initialize a layer of fractal modules.
        
        Args:
            dim: Feature dimension
            fractal_count: Number of fractal modules in this layer
            depth_range: Range of possible depths for fractals (min, max)
            dim_range: Range of possible fractal dimensions (min, max)
            dropout_prob: Dropout probability
        """
        super().__init__()
        
        self.fractals = nn.ModuleList([
            RecursiveProbabilisticFractal(
                input_dim=dim,
                output_dim=dim,
                depth=np.random.randint(depth_range[0], depth_range[1] + 1),
                fractal_dim=np.random.uniform(dim_range[0], dim_range[1]),
                dropout_prob=dropout_prob,
                seed=i  # Different seed for each fractal
            )
            for i in range(fractal_count)
        ])
        
        self.layer_norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all fractal modules."""
        # Apply each fractal module sequentially with residual connections
        for fractal in self.fractals:
            x = x + fractal(x)
        
        return self.layer_norm(x)