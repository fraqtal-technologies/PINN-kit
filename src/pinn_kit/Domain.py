import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
from skopt.space import Space
from skopt.sampler import Lhs
from skopt.sampler import Hammersly
from skopt.sampler import Halton
from skopt.sampler import Sobol


device = "mps"

class Domain:
    """A class representing a multi-dimensional domain for physics-informed neural networks.
    
    This class handles the definition of domain boundaries for arbitrary input variables and provides
    methods for sampling points within the domain using various sampling strategies.
    
    Args:
        variables (list): List of tuples (variable_name, min_value, max_value) defining the domain
                         Example: [('x', -1, 1), ('y', -1, 1), ('t', 0, 1)]
    """
    def __init__(self, variables):
        self.variables = variables
        self.variable_names = [var[0] for var in variables]
        self.bounds = [(var[1], var[2]) for var in variables]
        self.dimensions = len(variables)

    def sample_points(self, num_samples, sampler=None, fixed_values=None):
        """Sample points within the domain using specified sampling strategy.
        
        Args:
            num_samples (int): Number of points to sample
            sampler (str, optional): Sampling strategy to use. Options include:
                - None: Evenly spaced points
                - "random": Random uniform sampling
                - "lhs_classic": Classic Latin Hypercube Sampling
                - "lhs_centered": Centered Latin Hypercube Sampling
                - "halton": Halton sequence
                - "hammersly": Hammersly sequence
                - "sobol": Sobol sequence
            fixed_values (dict, optional): Dictionary mapping variable names to fixed values.
                                         If provided, those variables will be sampled at fixed values.
                                         Example: {'t': 0.5} will fix time to 0.5
            
        Returns:
            tuple: Arrays of sampled coordinates for each variable
        """
        if fixed_values is None:
            fixed_values = {}
        
        # Check if all fixed values correspond to valid variable names
        for var_name in fixed_values.keys():
            if var_name not in self.variable_names:
                raise ValueError(f"Variable '{var_name}' not found in domain variables: {self.variable_names}")
        
        # Determine which variables to sample and which are fixed
        sample_indices = []
        sample_bounds = []
        fixed_indices = []
        fixed_values_list = []
        
        for i, var_name in enumerate(self.variable_names):
            if var_name in fixed_values:
                fixed_indices.append(i)
                fixed_values_list.append(fixed_values[var_name])
            else:
                sample_indices.append(i)
                sample_bounds.append(self.bounds[i])
        
        # Generate evenly spaced samples
        if sampler is None:
            result_arrays = []
            for i in range(self.dimensions):
                if i in fixed_indices:
                    # Fixed value
                    fixed_idx = fixed_indices.index(i)
                    arr = np.ones([num_samples, 1]) * fixed_values_list[fixed_idx]
                else:
                    # Sample evenly spaced
                    sample_idx = sample_indices.index(i)
                    min_val, max_val = sample_bounds[sample_idx]
                    arr = np.linspace(min_val, max_val, num_samples).reshape(num_samples, 1)
                result_arrays.append(arr)
            
            return tuple(result_arrays)
        
        # Generate random samples
        elif sampler == "random":
            result_arrays = []
            for i in range(self.dimensions):
                if i in fixed_indices:
                    # Fixed value
                    fixed_idx = fixed_indices.index(i)
                    arr = np.ones([num_samples, 1]) * fixed_values_list[fixed_idx]
                else:
                    # Random uniform sampling
                    sample_idx = sample_indices.index(i)
                    min_val, max_val = sample_bounds[sample_idx]
                    arr = np.random.uniform(low=min_val, high=max_val, size=(num_samples, 1))
                result_arrays.append(arr)
            
            return tuple(result_arrays)
        
        # Generate samples using scikit-optimize samplers
        else:
            skip = 0
            # Latin Hypercube Sampling
            if sampler == "lhs_classic":
                sampler_obj = Lhs(lhs_type="classic")
            elif sampler == "lhs_centered":
                sampler_obj = Lhs(lhs_type="centered")
            elif sampler == "halton":
                skip = 1
                sampler_obj = Halton()
            elif sampler == "hammersly":
                skip = 1
                sampler_obj = Hammersly()
            elif sampler == "sobol":
                skip = 1
                sampler_obj = Sobol(randomize=False)
            else:
                print("This method is not available.")
                return None
            
            # Create space for sampling
            space = Space(sample_bounds)

            # Generate samples
            X = np.asarray(sampler_obj.generate(space.dimensions, num_samples + skip, random_state=10)[skip:])
            
            # Organize results
            result_arrays = []
            sample_idx = 0
            for i in range(self.dimensions):
                if i in fixed_indices:
                    # Fixed value
                    fixed_idx = fixed_indices.index(i)
                    arr = np.ones([num_samples, 1]) * fixed_values_list[fixed_idx]
                else:
                    # Sampled value
                    arr = X[:, sample_idx].reshape(num_samples, 1)
                    sample_idx += 1
                result_arrays.append(arr)
            
            return tuple(result_arrays)
    
    def sample_bc_points(self, fixed_points, num_samples, sampler=None):
        """Sample boundary condition points, including specific fixed points.
        
        Args:
            fixed_points (dict): Dictionary mapping variable names to fixed values
                               Example: {'x': 0, 'y': 0}
            num_samples (int): Total number of points to sample (including fixed points)
            sampler (str, optional): Sampling strategy to use for remaining points
            
        Returns:
            tuple: Arrays of sampled coordinates for each variable
        """
        # Sample points with fixed values
        result_arrays = self.sample_points(num_samples - 1, sampler=sampler, fixed_values=fixed_points)
        
        # Insert the fixed point at the beginning
        result_arrays = list(result_arrays)
        for i, var_name in enumerate(self.variable_names):
            if var_name in fixed_points:
                result_arrays[i] = np.insert(result_arrays[i], 0, fixed_points[var_name], axis=0)
        
        return tuple(result_arrays)
    
    # Backward compatibility methods for x, y, t
    @classmethod
    def from_xy_t(cls, x_min, x_max, y_min, y_max, t_min, t_max):
        """Create a Domain instance with x, y, t variables for backward compatibility.
        
        Args:
            x_min, x_max (float): x-coordinate bounds
            y_min, y_max (float): y-coordinate bounds  
            t_min, t_max (float): time bounds
            
        Returns:
            Domain: Domain instance with x, y, t variables
        """
        variables = [('x', x_min, x_max), ('y', y_min, y_max), ('t', t_min, t_max)]
        return cls(variables)
    
    def sample_points_xy_t(self, num_samples, sampler=None, fixed_time=None):
        """Backward compatibility method for sampling x, y, t points.
        
        Args:
            num_samples (int): Number of points to sample
            sampler (str, optional): Sampling strategy
            fixed_time (float, optional): Fixed time value
            
        Returns:
            tuple: (x_arr, y_arr, t_arr) Arrays of sampled coordinates
        """
        if self.variable_names != ['x', 'y', 't']:
            raise ValueError("This method only works with x, y, t domains. Use sample_points() for other domains.")
        
        fixed_values = {'t': fixed_time} if fixed_time is not None else None
        result = self.sample_points(num_samples, sampler=sampler, fixed_values=fixed_values)
        return result[0], result[1], result[2]  # x, y, t
    
    def sample_bc_points_xy_t(self, x0, y0, num_samples, sampler=None):
        """Backward compatibility method for sampling boundary condition points with x, y, t.
        
        Args:
            x0, y0 (float): Fixed x, y coordinates
            num_samples (int): Total number of points
            sampler (str, optional): Sampling strategy
            
        Returns:
            tuple: (x_arr, y_arr, t_arr) Arrays of sampled coordinates
        """
        if self.variable_names != ['x', 'y', 't']:
            raise ValueError("This method only works with x, y, t domains. Use sample_bc_points() for other domains.")
        
        fixed_points = {'x': x0, 'y': y0, 't': 0.0}
        result = self.sample_bc_points(fixed_points, num_samples, sampler=sampler)
        return result[0], result[1], result[2]  # x, y, t
    
# MODIFIED
def convert_to_torch_tensors(input_arrays):
    """Convert numpy arrays to PyTorch tensors.
    
    Args:
        input_arrays (list): List of numpy arrays to convert
        
    Returns:
        list: List of PyTorch tensors
    """
    input_tensors = []
    for input_array in input_arrays:
        input_tensors.append(torch.from_numpy(input_array).float())
    return input_tensors

def to_variable(domain):
    """Convert domain data to PyTorch variables with gradient tracking.
    
    Args:
        domain (list or numpy.ndarray): Input data to convert
        
    Returns:
        list or torch.Tensor: PyTorch variables with gradient tracking enabled
    """
    if isinstance(domain,list):
        domain_list_tensor=[]
        for i in domain:
            domain_list_tensor.append(Variable(torch.from_numpy(i.reshape(len(i),1)).float(),requires_grad=True).to(device))
        return domain_list_tensor
    if isinstance(domain,np.ndarray):
        return Variable(torch.from_numpy(domain).float(),requires_grad=True).to(device)
    

def convert_to_meshgrid(input_arrays):
    """Convert 1D arrays to a meshgrid and reshape to 2D arrays.
    
    This function takes a list of 1D arrays and creates a meshgrid from them,
    then reshapes the result to 2D arrays suitable for neural network input.
    
    Args:
        input_arrays (list): List of 1D numpy arrays to convert to meshgrid
                            Example: [x_arr, y_arr, t_arr] or [x_arr, y_arr]
        
    Returns:
        tuple: Reshaped meshgrid coordinates as 2D arrays
               Example: (xx, yy, tt) or (xx, yy) depending on input dimensions
    """
    if not isinstance(input_arrays, list):
        raise ValueError("input_arrays must be a list of numpy arrays")
    
    if len(input_arrays) < 2:
        raise ValueError("At least 2 arrays are required for meshgrid")
    
    # Calculate total number of samples
    num_samples = 1
    for arr in input_arrays:
        if not isinstance(arr, np.ndarray):
            raise ValueError("All inputs must be numpy arrays")
        num_samples *= arr.shape[0]
    
    # Create meshgrid
    meshgrid_arrays = np.meshgrid(*input_arrays)
    
    # Reshape each array to (num_samples, 1)
    result_arrays = []
    for arr in meshgrid_arrays:
        result_arrays.append(arr.reshape(num_samples, 1))
    
    return tuple(result_arrays)

def convert_to_meshgrid_xy_t(x_arr, y_arr, t_arr):
    """Backward compatibility function for convert_to_meshgrid with x, y, t parameters.
    
    This function maintains the old interface for existing code that uses
    the specific x, y, t parameter names.
    
    Args:
        x_arr (numpy.ndarray): 1D array of x-coordinates
        y_arr (numpy.ndarray): 1D array of y-coordinates
        t_arr (numpy.ndarray): 1D array of time values
        
    Returns:
        tuple: (xx, yy, tt) Reshaped meshgrid coordinates
    """
    return convert_to_meshgrid([x_arr, y_arr, t_arr])

def plot_points(x_arr,y_arr,title="Meshgrid of Points"):
    """Plot 2D points with specified title.
    
    Args:
        x_arr (numpy.ndarray): Array of x-coordinates
        y_arr (numpy.ndarray): Array of y-coordinates
        title (str, optional): Plot title. Defaults to "Meshgrid of Points"
    """
    # plot meshgrid points
    plt.scatter(x_arr, y_arr, color='blue', s=5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_time_points(t_arr):
    """Plot time points against a constant y-value of 1.
    
    Args:
        t_arr (numpy.ndarray): Array of time values
    """
    plot_points(t_arr,np.ones(t_arr.shape),title="Time Values")

# MODIFIED 

# extend pytorch dataset class so that we can use dataloader method
class DomainDataset(Dataset):
    """PyTorch Dataset class for domain data.
    
    This class extends PyTorch's Dataset class to handle domain data for use with DataLoader.
    
    Args:
        data (numpy.ndarray): Input data array
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        """Get the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        # returns the number of rows in data (the number of samples)
        return len(self.data)

    def __getitem__(self, idx):
        """Get a specific sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            numpy.ndarray: The requested sample
        """
        # returns the idx-th row of the data
        return self.data[idx]
    
def gen_batch_data(pt_domain,batch_size=10):
    """Generate batches of domain data using PyTorch DataLoader.
    
    Args:
        pt_domain (list): List of domain point arrays
        batch_size (int, optional): Size of each batch. Defaults to 10
        
    Returns:
        list: List of batched data tensors
    """
    data = np.hstack(tuple(pt_domain))
    dataset = DomainDataset(data)
    dataloader_list = list(DataLoader(dataset, batch_size=batch_size, shuffle=True))
    
    for i in range(len(dataloader_list)):
        dataloader_list[i] = dataloader_list[i].float()

    return dataloader_list