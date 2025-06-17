import numpy as np
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.Domain import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class PINN(nn.Module):
    """Physics-Informed Neural Network (PINN) implementation.
    
    This class implements a neural network that can be trained to solve differential equations
    by incorporating physical constraints into the loss function.
    
    Args:
        layer_list (list): List of integers specifying the number of neurons in each layer
        activation_function_name (str, optional): Name of activation function to use. Defaults to "tanh"
        initialisation_function_name (str, optional): Name of weight initialization function. Defaults to "xavier_normal"
        function_ansatz (callable, optional): Optional function to modify network output. Defaults to None
    """
    # initialises the network 
    def __init__(self, layer_list, activation_function_name = "tanh", initialisation_function_name = "xavier_normal", function_ansatz = None):
        super().__init__()
        self.num_inputs = layer_list[0]
        self.num_outputs = layer_list[len(layer_list)-1]
        self.num_layers = len(layer_list)-1
        self.linears = nn.ModuleList([nn.Linear(layer_list[i], layer_list[i+1]) for i in range(len(layer_list)-1)])
        self.activation_function_name = activation_function_name
        self.initialisation_function_name = initialisation_function_name
        self.function_ansatz = function_ansatz
        if initialisation_function_name is not None:
            self._init_weights()

    # initialise the weights in the neural network 
    def _init_weights(self,set_bias_to_zero=True):
        """Initialize the weights of the neural network layers.
        
        Args:
            set_bias_to_zero (bool, optional): Whether to initialize bias terms to zero. Defaults to True.
            
        Raises:
            ValueError: If unknown initialization function is specified
        """
        if self.initialisation_function_name == "xavier_normal":
            init_func = torch.nn.init.xavier_normal_
        elif self.initialisation_function_name == "xavier_uniform":
            init_func = torch.nn.init.xavier_uniform_
        else:
            raise ValueError(f'Unknown Initialisation Function: {self.initialisation_function_name}')
        for i in range(self.num_layers):
            # initialise the weights using xavier_normal function 
            init_func(self.linears[i].weight.data)
            if set_bias_to_zero == True:
                # set all the bias values to zero
                torch.nn.init.zeros_(self.linears[i].bias)

    # computes the forward pass with the defined neural network 
    # and its current weights
    def forward(self,input_tensors):
        """Perform forward pass through the neural network.
        
        Args:
            input_tensors (list): List of input tensors to be concatenated
            
        Returns:
            torch.Tensor: Network output after passing through all layers
            
        Raises:
            ValueError: If unknown activation function is specified
        """
        if self.activation_function_name == "tanh":
            act_func = torch.tanh
        elif self.activation_function_name == "relu":
            act_func = torch.relu
        elif self.activation_function_name == "sigmoid":
            act_func = torch.sigmoid
        elif self.activation_function_name == "soft_plus":
            act_func = softplus
        else:
            raise ValueError(f'Unknown Activation Function: {self.activation_function_name}')
        
        input = torch.cat(input_tensors, axis=1)
        for i, layer in enumerate(self.linears):
            if i == len(self.linears) - 1:
                softplus = nn.Softplus()
                input = act_func(layer(input))
            else:
                input = act_func(layer(input))

        if self.function_ansatz is not None:
            input = input * self.function_ansatz(input_tensors)
        
        return input
    
    def configure_optimiser(self, optimiser_name, initial_lr=0.002):
        """Configure the optimizer for training.
        
        Args:
            optimiser_name (str): Name of optimizer to use ("adam" or "lbfgs")
            initial_lr (float, optional): Initial learning rate. Defaults to 0.002
            
        Raises:
            ValueError: If unknown optimizer is specified
        """
        if optimiser_name == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(),
                                                lr=initial_lr)
            
        elif optimiser_name == "lbfgs":
            # default tolerance_grad = 1e-7
            # default tolerance_change = 1e-9
            self.optimizer = torch.optim.LBFGS(self.parameters(),
                                               lr=initial_lr,
                                               tolerance_grad = 1e-7,
                                               tolerance_change = 1e-9,
                                               line_search_fn = 'strong_wolfe')
        else:
            raise ValueError(f'Unknown Optimiser: {optimiser_name}')
        
    def configure_lr_scheduler(self,lr_scheduler_name=None,factor=0.5,patience=100):
        """Configure learning rate scheduler.
        
        Args:
            lr_scheduler_name (str, optional): Name of scheduler to use. Defaults to None
            factor (float, optional): Factor to reduce learning rate by. Defaults to 0.5
            patience (int, optional): Number of epochs to wait before reducing lr. Defaults to 100
        """
        if lr_scheduler_name == "reduce_lr_on_plateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=factor, patience=patience)
        else:
            self.scheduler = None 
    
    def set_path(self, path):
        """Set the path for saving model checkpoints.
        
        Args:
            path (str): Path where model checkpoints will be saved
        """
        self.path = path 

    def compute_loss(network, input_tensors, loss_list):
        """Compute the total loss for the network.
        
        Args:
            network (PINN): The neural network instance
            input_tensors (list): List of input tensors
            loss_list (list): List of dictionaries containing loss functions and their weights
            
        Returns:
            torch.Tensor: Total weighted loss
        """
        net_output = network(input_tensors)

        total_loss = 0 
        for loss_dict in loss_list:
            compute_loss_term = loss_dict["function"]
            loss_term = 0 
            if loss_dict["indices"] is not None:
                indices = loss_dict["indices"]
                loss_term = compute_loss_term(input_tensors, net_output, indices)
                total_loss += loss_dict["weight"]*compute_loss_term(input_tensors, net_output, indices)
                #print(compute_loss_term,loss_term)
            else:
                loss_term = compute_loss_term(input_tensors, net_output)
                total_loss += loss_dict["weight"]*compute_loss_term(input_tensors, net_output)
                #print(compute_loss_term,loss_term)
        return total_loss


    def train_model(self, input_arrays, loss_list, num_epochs, batch_size=None, monitoring_function=None):
        """Train the neural network.
        
        Args:
            input_arrays (list): List of input arrays
            loss_list (list): List of loss functions and their weights
            num_epochs (int): Number of training epochs
            batch_size (int, optional): Size of batches for training. Defaults to None
            monitoring_function (callable, optional): Function to monitor training progress. Defaults to None
            
        Returns:
            numpy.ndarray: Training history containing loss values
        """
        if isinstance(self.optimizer, torch.optim.Adam):
            return self._train_adam(input_arrays, loss_list, num_epochs, batch_size, monitoring_function=monitoring_function)
        elif isinstance(self.optimizer, torch.optim.LBFGS):
            return self._train_lbfgs(input_arrays, loss_list, num_epochs, monitoring_function=monitoring_function)

    def _train_lbfgs(self, input_arrays, loss_list, num_epochs, monitoring_function=None):
        """Train the network using L-BFGS optimizer.
        
        Args:
            input_arrays (list): List of input arrays
            loss_list (list): List of loss functions and their weights
            num_epochs (int): Number of training epochs
            monitoring_function (callable, optional): Function to monitor training progress. Defaults to None
            
        Returns:
            numpy.ndarray: Training history containing loss values
        """
        print("Training with LBFGS")

        tensor_list = convert_to_torch_tensors(input_arrays)
        input_tensors= []
        for input_tensor in tensor_list:
            input_tensors.append(Variable(input_tensor,requires_grad=True).to(device))

        # check what the initial loss is
        self.eval()
        
        testing_loss_list = []
        for i in range(len(loss_list)):
            testing_loss_list.append(loss_list[i].copy())
            if loss_list[i]["indices"] is not None:
                find_indices_function = loss_list[i]["indices"]
                testing_loss_list[i]["indices"] = find_indices_function(input_tensors)
        
        initial_loss = self.compute_loss(input_tensors, testing_loss_list)

        print("Initial Loss:",initial_loss.item())
        best_loss = initial_loss.item()
        self.train()

        training_history = np.zeros([num_epochs,])

        if monitoring_function is not None:
            monitored_value = monitoring_function(self)
            best_monitored_value = monitored_value

        # epoch specifies the number of times the model parameters will be updated
        for epoch in range(num_epochs):
            iter = 0 
            def closure():
                iter=0
                if torch.is_grad_enabled():     
                    # backward accumulates gradients, set the gradients to zero
                    self.optimizer.zero_grad() 
            
                # combining the loss functions
                loss = self.compute_loss(input_tensors, testing_loss_list)
                
                #print(iter)
                print("Iteration:",iter,"Loss:",loss.item())

                if loss.requires_grad:
                    loss.backward() 

                return loss  
            
            _ = self.optimizer.step(closure)

            updated_loss =  self.compute_loss(input_tensors, testing_loss_list)

            with torch.autograd.no_grad():
                training_history[epoch] = updated_loss.item()
                if updated_loss.item() < best_loss:
                    best_loss = updated_loss.item()  
                    torch.save(self.state_dict(),self.path+".pt")
                    print("Saving Best Loss:",best_loss)
                
                if monitoring_function is not None:
                    monitored_value = monitoring_function(self)
                    if monitored_value < best_monitored_value:
                        torch.save(self.state_dict(), self.path+"_best_monitored_value_model.pt")
                        best_monitored_value = monitored_value
    
                if self.scheduler is not None:
                    print("Epoch:",epoch,"Traning Loss:",updated_loss.item(),"LR:", self.scheduler.get_last_lr()) 
                else: 
                    print("Epoch:",epoch,"Traning Loss:",updated_loss.item())
        
        self.eval()
        final_loss = self.compute_loss(input_tensors, testing_loss_list)
        print("Final Loss:",final_loss.item())
        self.train()

        return training_history 


    def _train_adam(self, input_arrays, loss_list, num_epochs, batch_size, monitoring_function=None):
        """Train the network using Adam optimizer.
        
        Args:
            input_arrays (list): List of input arrays
            loss_list (list): List of loss functions and their weights
            num_epochs (int): Number of training epochs
            batch_size (int): Size of batches for training
            monitoring_function (callable, optional): Function to monitor training progress. Defaults to None
            
        Returns:
            numpy.ndarray: Training history containing loss values
        """
        print("Training with ADAM Optimiser")

        print(device)
        
        # will return a list containing each batch of data, each element is a tensor
        dataloader = gen_batch_data(input_arrays,batch_size=batch_size)

        num_batches = len(dataloader)
        print("Number of Batches:",num_batches)
        num_points_in_batch = input_arrays[0].shape[0]//num_batches

        initial_loss = 0

        # check what the initial loss is 
        self.eval()
        tensor_list = convert_to_torch_tensors(input_arrays)
        input_tensors= []
        for input_tensor in tensor_list:
            input_tensors.append(Variable(input_tensor,requires_grad=True).to(device))
        
        testing_loss_list = []
        for i in range(len(loss_list)):
            testing_loss_list.append(loss_list[i].copy())
            if loss_list[i]["indices"] is not None:
                find_indices_function = loss_list[i]["indices"]
                testing_loss_list[i]["indices"] = find_indices_function(input_tensors)
        
        initial_loss = self.compute_loss(input_tensors, testing_loss_list)
        
        self.train()

        print("Initial Total Loss: ",initial_loss.item())

        best_loss = initial_loss.item()
        training_history = np.zeros([num_epochs,])

        if monitoring_function is not None: 
            monitored_value = monitoring_function(self)
            best_monitored_value = monitored_value
        
        indices_list = []

        for epoch_idx in range(num_epochs):
            loss_for_epoch = 0

            for batch_idx in range(num_batches):
                # backward accumulates gradients, set the gradients to zero
                self.optimizer.zero_grad() 

                # load the data for this batch
                # get each column of the data (e.g. x, y and t)
                input_tensors_for_batch = []
                for i in range(self.num_inputs):
                    # get each column of the data (e.g. x, y and t)
                    input_tensor_for_batch = dataloader[batch_idx][:,i].reshape(num_points_in_batch,1)
                    input_tensors_for_batch.append(Variable(input_tensor_for_batch,requires_grad=True).to(device))

                if epoch_idx == 0:
                    # will store indices for each loss term for this batch 
                    batch_indices_list = []
                    for loss_dict in loss_list:
                        if loss_dict["indices"] is None:
                            batch_indices_list.append(None)
                        if loss_dict["indices"] is not None:
                            find_indices_function = loss_dict["indices"]
                            batch_indices_list.append(find_indices_function(input_tensors_for_batch))
                    # will store the indices for each batch 
                    # indices_list = [[indices for loss term 0 for batch 0, indices for loss term 1 for batch 0], [indices for loss term 0 for batch 1, indices for loss term 1 for batch 1]]
                    indices_list.append(batch_indices_list)

                loss_list_for_batch = []
                # for each loss term in loss_list
                for i in range(len(loss_list)):
                    loss_list_for_batch.append(loss_list[i].copy())
                    # add the actual indices to the loss term 
                    loss_list_for_batch[i]["indices"] = indices_list[batch_idx][i]
                
                # calculate the loss for this batch
                loss_for_batch = self.compute_loss(input_tensors_for_batch,loss_list_for_batch)

                # print the batch loss for each batch
                with torch.autograd.no_grad():
                    loss_for_epoch += loss_for_batch.item()
                    if self.scheduler is not None:
                                print("\r Epoch: ",epoch_idx,"Total Loss: ",loss_for_epoch/(batch_idx+1)," Batch Loss for Batch",batch_idx,": ",loss_for_batch.item(), " LR:", self.optimizer.param_groups[0]["lr"], end=' '*10)
                    else: 
                        print("\r Epoch: ",epoch_idx,"Total Loss: ",loss_for_epoch/(batch_idx+1)," Batch Loss for Batch",batch_idx,": ",loss_for_batch.item(), " LR:", self.optimizer.param_groups[0]["lr"], end=' '*10)
                        # print("\r Epoch: ",epoch_idx,"Total Loss: ",loss_for_epoch/(batch_idx+1)," Batch Loss for Batch",batch_idx,": ",loss_for_batch.item(),' '*10, end=' '*10)
                
                if num_batches == 1:
                    if loss_for_batch.item() < best_loss:
                        best_loss = loss_for_batch.item()  
                        torch.save(self.state_dict(), self.path+".pt")
                        print("\nSaving Best Loss (Total Loss): ",best_loss)

                # calculate the loss wrt weights for optimiser 
                loss_for_batch.backward() 

                # update neural network weights according to theta_new = theta_old - alpha * derivative of J w.r.t theta
                self.optimizer.step()
            
            training_history[epoch_idx] = loss_for_epoch/(batch_idx+1)
            
            print()
            
            # at the end of every epoch     
            # compute the loss of the saved model 
            # check final loss that the model is able to achieve
            if num_batches > 1:
                self.eval()
                val_loss = self.compute_loss(input_tensors,testing_loss_list).item()
                if val_loss < best_loss:
                    best_loss = val_loss 
                    torch.save(self.state_dict(), self.path+".pt")
                    print("Saving Best Loss (Total Loss): ",best_loss)
                self.train()
            
            if monitoring_function is not None: 
                self.eval()
                monitored_value = monitoring_function(self)
                if monitored_value < best_monitored_value:
                    torch.save(self.state_dict(), self.path+"_best_monitored_value_model.pt")
                    best_monitored_value = monitored_value
                self.train()
            
            if self.scheduler is not None:
                    self.scheduler.step(loss_for_epoch)

        # check final loss that the model is able to achieve
        self.eval()
        final_loss = initial_loss = self.compute_loss(input_tensors, testing_loss_list).item()
        print("Final Total Loss: ",final_loss)
        self.train()
        
        return training_history 

def plot_history(values,path=None):
    """Plot training history.
    
    Args:
        values (numpy.ndarray): Array of loss values to plot
        path (str, optional): Path to save the plot. Defaults to None
    """
    plt.figure(figsize=(10, 10))
    plt.plot(values, linestyle='-', color='b')
    plt.yscale('log')
    if path == None:
        plt.title('Loss History')
    else:
        plt.title('Loss History for '+path)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

def plot_3d_net_solution(net,epoch_idx,show_plot=False):
    """Plot 3D visualization of network solution.
    
    Args:
        net (PINN): Trained neural network
        epoch_idx (int): Current epoch number
        show_plot (bool, optional): Whether to display the plot. Defaults to False
        
    Returns:
        matplotlib.pyplot: Plot object if show_plot is False
    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    x_arr = np.linspace(X_MIN,X_MAX,100)
    y_arr = np.linspace(Y_MIN,Y_MAX,100)
    ms_x, ms_y = np.meshgrid(x_arr, y_arr)
    
    x_arr = ms_x.reshape(-1,1)
    y_arr = ms_y.reshape(-1,1)

    pt_x = Variable(torch.from_numpy(x_arr).float(), requires_grad=False).to(device)
    pt_y = Variable(torch.from_numpy(y_arr).float(), requires_grad=False).to(device)
    pt_t = 0*torch.ones_like(pt_x)

    net_output = net([pt_x,pt_y,pt_t])
    net_output = net_output.detach().cpu().numpy()
    ms_Q = net_output.reshape(ms_x.shape)

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_title('Predicted Solution after '+str(epoch_idx)+' Epochs of Training')
    #ax.set_zlim(0, 0.35)

    surf = ax.plot_surface(ms_x, ms_y, ms_Q, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5, location='left')

    if show_plot == True:
        plt.show()
    else:
        return plt