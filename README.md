# PINN-kit

A toolkit for Physics-Informed Neural Networks (PINNs). This package provides tools and utilities for implementing and training physics-informed neural networks for solving differential equations.

## Features

- Easy-to-use interface for defining physics-informed neural networks
- Support for various types of differential equations
- Domain handling utilities
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

## Quick Start

```python
from pinn_kit import PINN, Domain

# Create a domain
domain = Domain(...)

# Initialize a PINN
pinn = PINN(...)

# Define the loss terms
def loss():
    return loss_function

# Train the network
pinn.train_model(...)
```

## Documentation

For detailed documentation, please visit [our documentation page](https://github.com/shivani/PINN-kit).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
