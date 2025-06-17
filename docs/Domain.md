# Domain Module Documentation

The Domain module provides functionality for handling spatial-temporal domains in physics-informed neural networks, including point sampling, data conversion, and visualization.

## Class: Domain

A class representing a spatial-temporal domain with methods for sampling points using various strategies.

### Initialization

```python
Domain(x_min, x_max, y_min, y_max, t_min, t_max)
```

**Parameters:**
- `x_min` (float): Minimum x-coordinate of the domain
- `x_max` (float): Maximum x-coordinate of the domain
- `y_min` (float): Minimum y-coordinate of the domain
- `y_max` (float): Maximum y-coordinate of the domain
- `t_min` (float): Minimum time value
- `t_max` (float): Maximum time value

### Methods

#### sample_points
```python
sample_points(num_samples, sampler=None, fixed_time=None)
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
- `fixed_time` (float, optional): If provided, all points will be sampled at this time value

**Returns:**
- `tuple`: (x_arr, y_arr, t_arr) Arrays of sampled coordinates

#### sample_bc_points
```python
sample_bc_points(x0, y0, num_samples, sampler=None)
```
Sample boundary condition points, including a specific point (x0,y0).

**Parameters:**
- `x0` (float): x-coordinate of the fixed point
- `y0` (float): y-coordinate of the fixed point
- `num_samples` (int): Total number of points to sample (including fixed point)
- `sampler` (str, optional): Sampling strategy to use for remaining points

**Returns:**
- `tuple`: (x_arr, y_arr, t_arr) Arrays of sampled coordinates

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
convert_to_meshgrid(x_arr, y_arr, t_arr)
```
Convert 1D arrays to a 3D meshgrid and reshape to 2D arrays.

**Parameters:**
- `x_arr` (numpy.ndarray): 1D array of x-coordinates
- `y_arr` (numpy.ndarray): 1D array of y-coordinates
- `t_arr` (numpy.ndarray): 1D array of time values

**Returns:**
- `tuple`: (xx, yy, tt) Reshaped meshgrid coordinates

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

## Usage Example

```python
# Create a domain
domain = Domain(x_min=-1, x_max=1, y_min=-1, y_max=1, t_min=0, t_max=1)

# Sample points using Latin Hypercube Sampling
x_arr, y_arr, t_arr = domain.sample_points(
    num_samples=1000,
    sampler="lhs_classic"
)

# Convert to PyTorch tensors
tensors = convert_to_torch_tensors([x_arr, y_arr, t_arr])

# Create batches for training
batches = gen_batch_data([x_arr, y_arr, t_arr], batch_size=32)

# Plot the sampled points
plot_points(x_arr, y_arr, title="Sampled Points")
``` 