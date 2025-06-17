# PINN Module Documentation

The PINN (Physics-Informed Neural Network) module provides an implementation of neural networks that can be trained to solve differential equations by incorporating physical constraints into the loss function.

## Class: PINN

A PyTorch neural network implementation that combines traditional neural networks with physical constraints.

### Initialization

```python
PINN(layer_list, activation_function_name="tanh", initialisation_function_name="xavier_normal", function_ansatz=None)
```

**Parameters:**
- `layer_list` (list): List of integers specifying the number of neurons in each layer
- `activation_function_name` (str, optional): Name of activation function to use. Defaults to "tanh"
- `initialisation_function_name` (str, optional): Name of weight initialization function. Defaults to "xavier_normal"
- `function_ansatz` (callable, optional): Optional function to modify network output. Defaults to None

### Methods

#### _init_weights
```python
_init_weights(set_bias_to_zero=True)
```
Initializes the weights of the neural network layers.

**Parameters:**
- `set_bias_to_zero` (bool, optional): Whether to initialize bias terms to zero. Defaults to True.

**Raises:**
- `ValueError`: If unknown initialization function is specified

#### forward
```python
forward(input_tensors)
```
Performs forward pass through the neural network.

**Parameters:**
- `input_tensors` (list): List of input tensors to be concatenated

**Returns:**
- `torch.Tensor`: Network output after passing through all layers

**Raises:**
- `ValueError`: If unknown activation function is specified

#### configure_optimiser
```python
configure_optimiser(optimiser_name, initial_lr=0.002)
```
Configures the optimizer for training.

**Parameters:**
- `optimiser_name` (str): Name of optimizer to use ("adam" or "lbfgs")
- `initial_lr` (float, optional): Initial learning rate. Defaults to 0.002

**Raises:**
- `ValueError`: If unknown optimizer is specified

#### configure_lr_scheduler
```python
configure_lr_scheduler(lr_scheduler_name=None, factor=0.5, patience=100)
```
Configures learning rate scheduler.

**Parameters:**
- `lr_scheduler_name` (str, optional): Name of scheduler to use. Defaults to None
- `factor` (float, optional): Factor to reduce learning rate by. Defaults to 0.5
- `patience` (int, optional): Number of epochs to wait before reducing lr. Defaults to 100

#### set_path
```python
set_path(path)
```
Sets the path for saving model checkpoints.

**Parameters:**
- `path` (str): Path where model checkpoints will be saved

#### compute_loss
```python
compute_loss(network, input_tensors, loss_list)
```
Computes the total loss for the network.

**Parameters:**
- `network` (PINN): The neural network instance
- `input_tensors` (list): List of input tensors
- `loss_list` (list): List of dictionaries containing loss functions and their weights

**Returns:**
- `torch.Tensor`: Total weighted loss

#### train_model
```python
train_model(input_arrays, loss_list, num_epochs, batch_size=None, monitoring_function=None)
```
Trains the neural network.

**Parameters:**
- `input_arrays` (list): List of input arrays
- `loss_list` (list): List of loss functions and their weights
- `num_epochs` (int): Number of training epochs
- `batch_size` (int, optional): Size of batches for training. Defaults to None
- `monitoring_function` (callable, optional): Function to monitor training progress. Defaults to None

**Returns:**
- `numpy.ndarray`: Training history containing loss values

### Utility Functions

#### plot_history
```python
plot_history(values, path=None)
```
Plots training history.

**Parameters:**
- `values` (numpy.ndarray): Array of loss values to plot
- `path` (str, optional): Path to save the plot. Defaults to None

#### plot_3d_net_solution
```python
plot_3d_net_solution(net, epoch_idx, show_plot=False)
```
Plots 3D visualization of network solution.

**Parameters:**
- `net` (PINN): Trained neural network
- `epoch_idx` (int): Current epoch number
- `show_plot` (bool, optional): Whether to display the plot. Defaults to False

**Returns:**
- `matplotlib.pyplot`: Plot object if show_plot is False

## Usage Example

```python
# Create a PINN with 3 layers (2 inputs, 20 hidden neurons, 1 output)
layer_list = [2, 20, 1]
pinn = PINN(layer_list, activation_function_name="tanh")

# Configure optimizer
pinn.configure_optimiser("adam", initial_lr=0.001)

# Configure learning rate scheduler
pinn.configure_lr_scheduler("reduce_lr_on_plateau", factor=0.5, patience=100)

# Set path for saving checkpoints
pinn.set_path("model_checkpoints")

# Train the model
training_history = pinn.train_model(
    input_arrays=[x_data, y_data],
    loss_list=[{"function": loss_fn, "weight": 1.0}],
    num_epochs=1000,
    batch_size=32
)

# Plot training history
plot_history(training_history)
``` 