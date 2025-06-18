# Domain Module Documentation

The Domain module provides functionality for handling multi-dimensional domains in physics-informed neural networks, including point sampling, data conversion, and visualization. The Domain class now supports arbitrary input variables, not just x, y, and t.

## Class: Domain

A class representing a multi-dimensional domain with methods for sampling points using various strategies.

### Initialization

The Domain class now accepts an arbitrary list of input variables:

```python
Domain(variables)
```

**Parameters:**
- `variables` (list): List of tuples (variable_name, min_value, max_value) defining the domain
  - `variable_name` (str): Name of the variable (e.g., 'x', 'y', 't', 'param')
  - `min_value` (float): Minimum value for the variable
  - `max_value` (float): Maximum value for the variable

**Examples:**
```python
# 2D spatial domain
domain_2d = Domain([('x', -1, 1), ('y', -1, 1)])

# 3D spatial-temporal domain
domain_3d = Domain([('x', -5, 5), ('y', -5, 5), ('t', 0, 1)])

# 4D domain with additional parameters
domain_4d = Domain([('x', -1, 1), ('y', -1, 1), ('t', 0, 1), ('param', 0, 10)])
```

### Backward Compatibility

For existing code that uses x, y, t variables, you can use the convenience method:

```python
Domain.from_xy_t(x_min, x_max, y_min, y_max, t_min, t_max)
```

**Parameters:**
- `x_min, x_max` (float): x-coordinate bounds
- `y_min, y_max` (float): y-coordinate bounds  
- `t_min, t_max` (float): time bounds

**Example:**
```python
# Equivalent to the old constructor
domain = Domain.from_xy_t(x_min=-1, x_max=1, y_min=-1, y_max=1, t_min=0, t_max=1)
```

### Methods

#### sample_points
```python
sample_points(num_samples, sampler=None, fixed_values=None)
```
Sample points within the domain using specified sampling strategy.

**Parameters:**
- `num_samples` (int): Number of points to sample
- `sampler` (str, optional): Sampling strategy to use. Options include:
  - None: Evenly spaced points
  - "random": Random uniform sampling
  - "lhs_classic": Classic Latin Hypercube Sampling
  - "lhs_centered": Centered Latin Hypercube Sampling
  - "halton": Halton sequence
  - "hammersly": Hammersly sequence
  - "sobol": Sobol sequence
- `fixed_values` (dict, optional): Dictionary mapping variable names to fixed values.
  If provided, those variables will be sampled at fixed values.
  Example: {'t': 0.5} will fix time to 0.5

**Returns:**
- `tuple`: Arrays of sampled coordinates for each variable

**Examples:**
```python
# Sample all variables
x_arr, y_arr, t_arr = domain.sample_points(num_samples=100)

# Sample with fixed time
x_arr, y_arr, t_arr = domain.sample_points(num_samples=100, fixed_values={'t': 0.5})

# Sample with multiple fixed values
x_arr, y_arr, t_arr, param_arr = domain_4d.sample_points(
    num_samples=100, 
    fixed_values={'t': 0.5, 'param': 5.0}
)
```

#### sample_bc_points
```python
sample_bc_points(fixed_points, num_samples, sampler=None)
```
Sample boundary condition points, including specific fixed points.

**Parameters:**
- `fixed_points` (dict): Dictionary mapping variable names to fixed values
  Example: {'x': 0, 'y': 0}
- `num_samples` (int): Total number of points to sample (including fixed points)
- `sampler` (str, optional): Sampling strategy to use for remaining points

**Returns:**
- `tuple`: Arrays of sampled coordinates for each variable

**Example:**
```python
# Sample boundary conditions with fixed x, y
x_arr, y_arr, t_arr = domain.sample_bc_points(
    fixed_points={'x': 0, 'y': 0}, 
    num_samples=100
)
```

#### Backward Compatibility Methods

For existing code that expects x, y, t specific methods:

```python
sample_points_xy_t(num_samples, sampler=None, fixed_time=None)
```
Backward compatibility method for sampling x, y, t points.

```python
sample_bc_points_xy_t(x0, y0, num_samples, sampler=None)
```
Backward compatibility method for sampling boundary condition points with x, y, t.

**Example:**
```python
# Old-style usage (still works)
domain = Domain.from_xy_t(-1, 1, -1, 1, 0, 1)
x_arr, y_arr, t_arr = domain.sample_points_xy_t(100, fixed_time=0.5)
x_arr, y_arr, t_arr = domain.sample_bc_points_xy_t(0, 0, 100)
```

## Utility Functions

### convert_to_torch_tensors
```python
convert_to_torch_tensors(input_arrays)
```
Convert numpy arrays to PyTorch tensors.

**Parameters:**
- `input_arrays` (list): List of numpy arrays to convert

**Returns:**
- `list`: List of PyTorch tensors

### to_variable
```python
to_variable(domain)
```
Convert domain data to PyTorch variables with gradient tracking.

**Parameters:**
- `domain` (list or numpy.ndarray): Input data to convert

**Returns:**
- `list or torch.Tensor`: PyTorch variables with gradient tracking enabled

### convert_to_meshgrid
```python
convert_to_meshgrid(input_arrays)
```
Convert 1D arrays to a meshgrid and reshape to 2D arrays.

This function takes a list of 1D arrays and creates a meshgrid from them,
then reshapes the result to 2D arrays suitable for neural network input.

**Parameters:**
- `input_arrays` (list): List of 1D numpy arrays to convert to meshgrid
  Example: [x_arr, y_arr, t_arr] or [x_arr, y_arr]

**Returns:**
- `tuple`: Reshaped meshgrid coordinates as 2D arrays
  Example: (xx, yy, tt) or (xx, yy) depending on input dimensions

**Examples:**
```python
# 2D meshgrid
x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 10)
xx, yy = convert_to_meshgrid([x, y])

# 3D meshgrid
t = np.linspace(0, 1, 5)
xx, yy, tt = convert_to_meshgrid([x, y, t])

# 4D meshgrid (if needed)
z = np.linspace(0, 1, 3)
xx, yy, tt, zz = convert_to_meshgrid([x, y, t, z])
```

### convert_to_meshgrid_xy_t
```python
convert_to_meshgrid_xy_t(x_arr, y_arr, t_arr)
```
Backward compatibility function for convert_to_meshgrid with x, y, t parameters.

This function maintains the old interface for existing code that uses
the specific x, y, t parameter names.

**Parameters:**
- `x_arr` (numpy.ndarray): 1D array of x-coordinates
- `y_arr` (numpy.ndarray): 1D array of y-coordinates
- `t_arr` (numpy.ndarray): 1D array of time values

**Returns:**
- `tuple`: (xx, yy, tt) Reshaped meshgrid coordinates

**Example:**
```python
# Old interface (still works)
xx, yy, tt = convert_to_meshgrid_xy_t(x, y, t)
```

### plot_points
```python
plot_points(x_arr, y_arr, title="Meshgrid of Points")
```
Plot 2D points with specified title.

**Parameters:**
- `x_arr` (numpy.ndarray): Array of x-coordinates
- `y_arr` (numpy.ndarray): Array of y-coordinates
- `title` (str, optional): Plot title. Defaults to "Meshgrid of Points"

### plot_time_points
```python
plot_time_points(t_arr)
```
Plot time points against a constant y-value of 1.

**Parameters:**
- `t_arr` (numpy.ndarray): Array of time values

## Class: DomainDataset

A PyTorch Dataset class for handling domain data.

### Initialization
```python
DomainDataset(data)
```

**Parameters:**
- `data` (numpy.ndarray): Input data array

### Methods

#### __len__
```python
__len__()
```
Get the number of samples in the dataset.

**Returns:**
- `int`: Number of samples

#### __getitem__
```python
__getitem__(idx)
```
Get a specific sample from the dataset.

**Parameters:**
- `idx` (int): Index of the sample to retrieve

**Returns:**
- `numpy.ndarray`: The requested sample

### gen_batch_data
```python
gen_batch_data(pt_domain, batch_size=10)
```
Generate batches of domain data using PyTorch DataLoader.

**Parameters:**
- `pt_domain` (list): List of domain point arrays
- `batch_size` (int, optional): Size of each batch. Defaults to 10

**Returns:**
- `list`: List of batched data tensors

## Usage Examples

### New Flexible Interface

```python
# Create a 3D domain
domain = Domain([('x', -1, 1), ('y', -1, 1), ('t', 0, 1)])

# Sample points using Latin Hypercube Sampling
x_arr, y_arr, t_arr = domain.sample_points(
    num_samples=1000,
    sampler="lhs_classic"
)

# Sample with fixed time
x_arr, y_arr, t_arr = domain.sample_points(
    num_samples=1000,
    sampler="lhs_classic",
    fixed_values={'t': 0.5}
)

# Sample boundary conditions
x_arr, y_arr, t_arr = domain.sample_bc_points(
    fixed_points={'x': 0, 'y': 0},
    num_samples=100
)

# Convert to PyTorch tensors
tensors = convert_to_torch_tensors([x_arr, y_arr, t_arr])

# Create meshgrid from 1D arrays
x_1d = np.linspace(-1, 1, 50)
y_1d = np.linspace(-1, 1, 50)
t_1d = np.linspace(0, 1, 10)

# Create 3D meshgrid
xx, yy, tt = convert_to_meshgrid([x_1d, y_1d, t_1d])

# Create 2D meshgrid (for spatial problems)
xx_2d, yy_2d = convert_to_meshgrid([x_1d, y_1d])

# Create batches for training
batches = gen_batch_data([x_arr, y_arr, t_arr], batch_size=32)

# Plot the sampled points
plot_points(x_arr, y_arr, title="Sampled Points")
```

### Backward Compatibility

```python
# Create domain using old-style constructor
domain = Domain.from_xy_t(x_min=-1, x_max=1, y_min=-1, y_max=1, t_min=0, t_max=1)

# Use old-style sampling methods
x_arr, y_arr, t_arr = domain.sample_points_xy_t(
    num_samples=1000,
    sampler="lhs_classic"
)

x_arr, y_arr, t_arr = domain.sample_bc_points_xy_t(0, 0, 100)

# Use old-style meshgrid function
xx, yy, tt = convert_to_meshgrid_xy_t(x_1d, y_1d, t_1d)
```