# PINN-kit 🧠⚡

A comprehensive toolkit for **Physics-Informed Neural Networks (PINNs)** that empowers researchers and engineers to solve complex differential equations using deep learning.

!!! abstract "What is PINN-kit?"
    PINN-kit provides an intuitive interface for implementing and training physics-informed neural networks, 
    combining the power of deep learning with the rigor of physics-based modeling.

## ✨ Key Features

- 🎯 **Easy-to-use interface** for defining physics-informed neural networks
- 🔧 **Flexible domain handling** utilities for arbitrary input variables  
- 🚀 **High-performance training** with PyTorch backend
- 📊 **Built-in visualization** tools for results analysis
- 🔬 **Support for various** differential equation types
- 🎨 **Clean, modular design** for easy customization

## 🚀 Quick Start

### Installation

PINN-kit supports macOS, Linux, and Windows. Install it using pip:

```bash
pip install pinn-kit
```

!!! tip "Virtual Environment"
    It's recommended to use a virtual environment with Python 3.12+ to avoid dependency conflicts:
    
    ```bash
    python3.12 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install pinn-kit
    ```

### Basic Usage

```python
from pinn_kit import PINN, Domain

# Create a domain with flexible variable definition
domain = Domain([
    ('x', -1, 1),  # x-coordinate bounds
    ('y', -1, 1),  # y-coordinate bounds  
    ('t', 0, 1)    # time bounds
])

# Initialize a PINN with 3 inputs, 2 hidden layers, 1 output
pinn = PINN([3, 20, 20, 1])

# Define your physics-informed loss function
def physics_loss():
    # Your physics constraints here
    return loss_value

# Train the network
pinn.train_model(domain, physics_loss, epochs=1000)
```

!!! success "That's it!"
    PINN-kit handles all the complexity behind the scenes, letting you focus on your physics problem.

## 📚 Documentation

Explore our comprehensive documentation to get the most out of PINN-kit:

=== "Usage Guide"
    Learn how to use PINN-kit effectively with step-by-step tutorials and examples.

=== "Domain Module" 
    Understand how to define and work with different types of domains and boundaries.

=== "PINN Module"
    Dive deep into the PINN implementation and advanced configuration options.

=== "Examples"
    See real-world applications and use cases with complete code examples.

## 🎯 Use Cases

PINN-kit is perfect for solving:

- **Heat transfer** and **fluid dynamics** problems
- **Wave propagation** and **vibration analysis**
- **Quantum mechanics** and **electromagnetic field** simulations
- **Financial modeling** with differential constraints
- **Biological systems** and **chemical reactions**

## 🔧 System Requirements

| Component | Requirement |
|-----------|-------------|
| Python    | 3.9+        |
| PyTorch   | 2.0+        |
| NumPy     | 1.21+       |
| OS        | macOS, Linux, Windows |

!!! note "Dependencies"
    All required dependencies (including PyTorch) are automatically installed with PINN-kit.

## 🤝 Contributing

We welcome contributions! Whether it's:

- 🐛 **Bug reports** and feature requests
- 📝 **Documentation** improvements  
- 💻 **Code contributions** and pull requests
- 💡 **Ideas** and suggestions

Visit our [GitHub repository](https://github.com/shivani/PINN-kit) to get started.

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with ❤️ for the scientific computing community**

[![GitHub stars](https://img.shields.io/github/stars/shivani/PINN-kit?style=social)](https://github.com/shivani/PINN-kit)
[![PyPI version](https://badge.fury.io/py/pinn-kit.svg)](https://pypi.org/project/pinn-kit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>
