# PINN-kit

A toolkit for Physics-Informed Neural Networks (PINNs). This package provides tools and utilities for implementing and training physics-informed neural networks for solving differential equations.

## Features

- Easy-to-use interface for defining physics-informed neural networks
- Support for various types of differential equations
- Flexible domain handling utilities for arbitrary input variables
- Advanced meshgrid functionality for multi-dimensional problems
- Training and evaluation tools

## Installation

PINN-kit supports macOS, Linux, and Windows. You can install it using pip, which will also automatically install all required dependencies, including torch.

### Create a Virtual Environment (Recommended)

It is recommended to use a virtual environment with Python 3.12 to avoid dependency conflicts.

#### macOS & Linux

```bash
python3.12 -m venv venv
source venv/bin/activate
```

#### Windows

Open Command Prompt or PowerShell and run:

```bash
python3.12 -m venv venv
venv\Scripts\activate
```

### Install PINN-kit

#### macOS & Linux

```bash
pip install pinn-kit
```

#### Windows

Open Command Prompt or PowerShell and run:

```bash
pip install pinn-kit
```

> **Note:** There is no need to install torch separately; it will be installed automatically with pinn-kit.

## PINN-kit is easy to use

```python
from pinn_kit import PINN, Domain, convert_to_meshgrid
import numpy as np

# Create a domain with flexible variable definition
domain = Domain([
    ('x', -1, 1),  # x-coordinate bounds
    ('y', -1, 1),  # y-coordinate bounds
    ('t', 0, 1)    # time bounds
])

# Sample points using various strategies
x_arr, y_arr, t_arr = domain.sample_points(
    num_samples=1000,
    sampler="lhs_classic"  # Latin Hypercube Sampling
)

# Create meshgrid from 1D arrays (new flexible interface)
x_1d = np.linspace(-1, 1, 50)
y_1d = np.linspace(-1, 1, 50)
t_1d = np.linspace(0, 1, 10)
xx, yy, tt = convert_to_meshgrid([x_1d, y_1d, t_1d])

# Initialize a PINN
pinn = PINN([3, 20, 20, 1])  # 3 inputs, 2 hidden layers, 1 output

# Define the loss terms
def loss():
    return loss_function

# Train the network
pinn.train_model([x_arr, y_arr, t_arr], loss_list, num_epochs=1000)
```

## Documentation

For detailed documentation, please visit [our documentation page](https://github.com/shivani/PINN-kit).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
