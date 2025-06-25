# LLM Rules for Converting LaTeX Differential Equations to PINN Loss Functions

This guide provides a set of rules for Large Language Models (LLMs) to convert differential equations (inputted in LaTeX) into loss functions compatible with the pinn-kit library. These rules enable automated generation of PyTorch code for PINN residual and boundary/initial losses.

---

## 1. Parse the LaTeX Differential Equation
- Identify the main equation and all variables.
- Recognize derivatives (e.g., $\frac{d^2u}{dx^2}$, $\frac{\partial u}{\partial t}$).
- Extract boundary and/or initial conditions.

## 2. Map Variables to PyTorch Tensors
- Each independent variable (e.g., $x$, $t$) corresponds to an input tensor (e.g., `x = input_tensors[0]`).
- The network output (e.g., $u$ or $x$) is the predicted solution (`u` or `x`).

## 3. Translate Derivatives
- Use `torch.autograd.grad` to compute derivatives:
  - First derivative: `u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]`
  - Second derivative: `u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]`
- For higher-order or mixed derivatives, apply `autograd.grad` recursively.

## 4. Construct the Residual Function
- Rearrange the equation so that the right-hand side is zero (e.g., $\mathcal{F}(u, x) = 0$).
- Implement the residual as a function returning the left-hand side minus the right-hand side.
- Example (Poisson):
  ```python
  def compute_residual(input_tensors, u):
      x = input_tensors[0]
      u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
      u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
      residual = -u_xx - (np.pi**2) * torch.sin(np.pi * x)
      return residual
  ```

## 5. Construct Loss Functions
- **Residual Loss:**
  - Compare the residual to zero using MSE loss.
  - Example:
    ```python
    def residual_loss(input_tensors, net_output):
        pred_residual_values = compute_residual(input_tensors, net_output)
        true_residual_values = torch.zeros_like(pred_residual_values)
        return torch.nn.MSELoss()(pred_residual_values, true_residual_values)
    ```
- **Boundary/Initial Loss:**
  - Identify indices where boundary/initial conditions apply.
  - Compare network output (or its derivatives) to the specified value using MSE loss.
  - Example (Dirichlet):
    ```python
    def boundary_loss(input_tensors, net_output, boundary_indices):
        pred_boundary_values = net_output[boundary_indices]
        true_boundary_values = torch.zeros_like(pred_boundary_values)
        return torch.nn.MSELoss()(pred_boundary_values, true_boundary_values)
    ```
  - Example (Initial velocity):
    ```python
    def initial_velocity_loss(input_tensors, net_output):
        t = input_tensors[0]
        initial_indices = torch.nonzero(t[:, 0] == T_MIN)
        x_initial = net_output[initial_indices]
        x_t = torch.autograd.grad(x_initial, t[initial_indices], grad_outputs=torch.ones_like(x_initial), create_graph=True)[0]
        true_velocity = torch.zeros_like(x_t)
        return torch.nn.MSELoss()(x_t, true_velocity)
    ```

## 6. Example Conversion

### Example 1: 1D Poisson Equation
LaTeX:
$$
-\frac{d^2u}{dx^2} = \pi^2 \sin(\pi x), \quad u(-1) = 0, \ u(1) = 0
$$

- Residual: `-u_xx - (np.pi**2) * torch.sin(np.pi * x)`
- Boundary: `u(-1) = 0`, `u(1) = 0`

### Example 2: Harmonic Oscillator ODE
LaTeX:
$$
\frac{d^2x}{dt^2} + \omega^2 x = 0, \quad x(0) = 1, \ \frac{dx}{dt}(0) = 0
$$

- Residual: `x_tt + (omega**2) * x`
- Initial: `x(0) = 1`, `x'(0) = 0`

---

## 7. General Workflow for LLMs
1. Receive the LaTeX equation and conditions.
2. Parse and identify variables, derivatives, and conditions.
3. Generate PyTorch code for:
   - Residual function
   - Residual loss
   - Boundary/initial loss functions
4. Output code snippets ready to be used in pinn-kit training scripts.

---

For more details, see the [PINN-kit documentation](https://github.com/shivani/PINN-kit). 