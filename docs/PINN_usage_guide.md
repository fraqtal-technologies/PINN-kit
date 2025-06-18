# PINN-kit Usage Guide

This guide explains how to use the PINN-kit library to develop and train Physics-Informed Neural Networks (PINNs) for solving differential equations.

## Table of Contents
1. [Setup](#setup)
2. [Domain Definition](#domain-definition)
3. [Network Architecture](#network-architecture)
4. [Loss Functions](#loss-functions)
5. [Training Process](#training-process)
6. [Example Usage](#example-usage)

## Setup

First, ensure you have the required dependencies installed and import the package as follows:

```python
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pinn_kit import Domain, PINN
```

## Domain Definition

The `Domain` class helps define your problem's spatial and temporal boundaries. The Domain class now supports arbitrary input variables, not just x, y, and t.

### New Flexible Interface

```python
# Define domain with arbitrary variables
domain = Domain([
    ('x', -5.0, 5.0),   # x-coordinate bounds
    ('y', -5.0, 5.0),   # y-coordinate bounds
    ('t', 0.0, 1.0)     # time bounds
])

# For 2D problems (no time)
domain_2d = Domain([
    ('x', -1.0, 1.0),
    ('y', -1.0, 1.0)
])

# For problems with additional parameters
domain_with_params = Domain([
    ('x', -1.0, 1.0),
    ('y', -1.0, 1.0),
    ('t', 0.0, 1.0),
    ('param', 0.0, 10.0)  # additional parameter
])
```

### Backward Compatibility

For existing code that uses x, y, t variables:

```python
# Define domain boundaries (old style)
domain = Domain.from_xy_t(
    x_min=-5.0,  # minimum x-coordinate
    x_max=5.0,   # maximum x-coordinate
    y_min=-5.0,  # minimum y-coordinate
    y_max=5.0,   # maximum y-coordinate
    t_min=0.0,   # minimum time
    t_max=1.0    # maximum time
)
```

### Sampling Points

You can sample points within your domain using various strategies:

```python
# Evenly spaced points
x_arr, y_arr, t_arr = domain.sample_points(num_samples=100)

# Random sampling
x_arr, y_arr, t_arr = domain.sample_points(num_samples=100, sampler="random")

# Latin Hypercube Sampling
x_arr, y_arr, t_arr = domain.sample_points(num_samples=100, sampler="lhs_classic")

# Sample with fixed time
x_arr, y_arr, t_arr = domain.sample_points(
    num_samples=100, 
    sampler="lhs_classic",
    fixed_values={'t': 0.5}
)

# Other available samplers:
# - "lhs_centered": Centered Latin Hypercube Sampling
# - "halton": Halton sequence
# - "hammersly": Hammersly sequence
# - "sobol": Sobol sequence
```

### Backward Compatibility Sampling

For existing code that expects the old interface:

```python
# Old-style sampling (still works)
x_arr, y_arr, t_arr = domain.sample_points_xy_t(num_samples=100, fixed_time=0.5)
```

## Network Architecture

The `PINN` class implements the neural network architecture:

```python
# Define network architecture
layer_list = [3] + [20] * 9 + [1]  # 3 inputs, 9 hidden layers with 20 neurons each, 1 output
network = PINN(
    layer_list,
    activation_function_name="tanh",  # Options: "tanh", "relu", "sigmoid", "soft_plus"
    initialisation_function_name="xavier_normal"  # Options: "xavier_normal", "xavier_uniform"
)
```

## Loss Functions

You need to define three types of loss functions:

1. **Residual Loss**: Enforces the differential equation
2. **Initial Condition Loss**: Enforces initial conditions
3. **Boundary Condition Loss**: Enforces boundary conditions

Example loss function definitions:

```python
def compute_residual(input_tensors, Q):
    x, y, t = input_tensors[0], input_tensors[1], input_tensors[2]
    
    # Compute derivatives
    Q_x = torch.autograd.grad(Q.sum(), x, create_graph=True)[0]
    Q_y = torch.autograd.grad(Q.sum(), y, create_graph=True)[0]
    Q_t = torch.autograd.grad(Q.sum(), t, create_graph=True)[0]
    
    # Define your differential equation here
    residual = Q_t - (Q_xx + Q_yy)  # Example: Heat equation
    
    return residual

def compute_ic_loss(input_tensors, net_output, ic_indices):
    # Compute initial condition loss
    pred_ic_values = net_output[ic_indices]
    true_ic_values = compute_true_ic_values(...)  # Define your initial condition
    return torch.nn.MSELoss()(pred_ic_values, true_ic_values)

def compute_bc_loss(input_tensors, net_output, boundary_indices):
    # Compute boundary condition loss
    pred_bc_values = net_output[boundary_indices]
    true_bc_values = compute_true_bc_values(...)  # Define your boundary condition
    return torch.nn.MSELoss()(pred_bc_values, true_bc_values)
```

## Training Process

Configure the training process:

```python
# Configure optimizer
network.configure_optimiser(
    optimiser_name="adam",  # Options: "adam", "lbfgs"
    initial_lr=0.002
)

# Configure learning rate scheduler (optional)
network.configure_lr_scheduler(
    lr_scheduler_name="reduce_lr_on_plateau",
    factor=0.5,
    patience=100
)

# Define loss functions list
loss_list = [
    {"function": residual_loss, "indices": None, "weight": 1},
    {"function": ic_loss, "indices": find_ic_indices, "weight": 1},
    {"function": bc_loss, "indices": find_boundary_indices, "weight": 1}
]

# Train the model
training_history = network.train_model(
    input_arrays=[x_arr, y_arr, t_arr],
    loss_list=loss_list,
    num_epochs=1000,
    batch_size=125000,
    monitoring_function=None  # Optional monitoring function
)
```

## Example Usage

Here's a complete example of solving a 2D heat equation using the new flexible interface:

```python
# 1. Define domain using new flexible interface
domain = Domain([
    ('x', -5, 5),  # x-coordinate bounds
    ('y', -5, 5),  # y-coordinate bounds
    ('t', 0, 1)    # time bounds
])

# 2. Sample points
x_arr, y_arr, t_arr = domain.sample_points(num_samples=49)

# 3. Create network
layer_list = [3] + [20] * 9 + [1]
network = PINN(layer_list)

# 4. Configure training
network.configure_optimiser("adam")
network.set_path("model_checkpoints")

# 5. Define loss functions
def compute_residual(input_tensors, Q):
    x, y, t = input_tensors[0], input_tensors[1], input_tensors[2]
    Q_x = torch.autograd.grad(Q.sum(), x, create_graph=True)[0]
    Q_y = torch.autograd.grad(Q.sum(), y, create_graph=True)[0]
    Q_t = torch.autograd.grad(Q.sum(), t, create_graph=True)[0]
    Q_xx = torch.autograd.grad(Q_x.sum(), x, create_graph=True)[0]
    Q_yy = torch.autograd.grad(Q_y.sum(), y, create_graph=True)[0]
    return Q_t - (Q_xx + Q_yy)

# 6. Train model
loss_list = [
    {"function": compute_residual, "indices": None, "weight": 1},
    {"function": ic_loss, "indices": find_ic_indices, "weight": 1},
    {"function": bc_loss, "indices": find_boundary_indices, "weight": 1}
]

training_history = network.train_model(
    input_arrays=[x_arr, y_arr, t_arr],
    loss_list=loss_list,
    num_epochs=1000,
    batch_size=125000
)
```

### Example with Additional Parameters

```python
# Define domain with additional parameters
domain = Domain([
    ('x', -1, 1),
    ('y', -1, 1),
    ('t', 0, 1),
    ('param', 0, 10)  # additional parameter
])

# Sample points with fixed parameter value
x_arr, y_arr, t_arr, param_arr = domain.sample_points(
    num_samples=100,
    fixed_values={'param': 5.0}
)

# Use in network (adjust layer_list for 4 inputs)
layer_list = [4] + [20] * 9 + [1]  # 4 inputs instead of 3
network = PINN(layer_list)

# Train with 4 input arrays
training_history = network.train_model(
    input_arrays=[x_arr, y_arr, t_arr, param_arr],
    loss_list=loss_list,
    num_epochs=1000,
    batch_size=125000
)
```

### Backward Compatibility Example

```python
# Use old-style interface (still works)
domain = Domain.from_xy_t(x_min=-5, x_max=5, y_min=-5, y_max=5, t_min=0, t_max=1)

# Old-style sampling
x_arr, y_arr, t_arr = domain.sample_points_xy_t(num_samples=49)

# Rest of the workflow remains the same
layer_list = [3] + [20] * 9 + [1]
network = PINN(layer_list)
# ... continue with training
```

## Tips for Success

1. **Domain Size**: Choose appropriate domain boundaries that capture the physical behavior you're interested in.

2. **Sampling Strategy**: 
   - Use evenly spaced points for simple problems
   - Use Latin Hypercube Sampling for better coverage in complex domains
   - Consider using more points in regions where the solution changes rapidly

3. **Network Architecture**:
   - Start with a moderate number of layers (5-10) and neurons (20-50)
   - Use tanh activation for smooth solutions
   - Use ReLU for solutions with sharp gradients

4. **Training**:
   - Start with Adam optimizer for most problems
   - Use L-BFGS for problems with smooth solutions
   - Monitor the loss components separately to identify which constraints are harder to satisfy
   - Use learning rate scheduling if the loss plateaus

5. **Loss Weights**:
   - Adjust the weights of different loss components if one type of constraint is harder to satisfy
   - Consider using adaptive weights that change during training

## Common Issues and Solutions

1. **Training Instability**:
   - Reduce learning rate
   - Use gradient clipping
   - Normalize input data

2. **Poor Solution Accuracy**:
   - Increase number of training points
   - Add more layers or neurons
   - Try different sampling strategies

3. **Slow Training**:
   - Reduce batch size
   - Use GPU acceleration
   - Simplify network architecture

4. **Overfitting**:
   - Add regularization
   - Reduce network size
   - Increase number of training points 