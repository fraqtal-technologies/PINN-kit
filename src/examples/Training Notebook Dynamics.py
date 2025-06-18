#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from Domain import *
from PINN import *
import json


# In[2]:


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
        )

print(f"Using {device} device")


# ## 1. Define Model Specific Functions

# In[3]:


gamma = 0.4
n_bar = 2

calculate_mse = torch.nn.MSELoss() 

def compute_true_ic_values(pt_ic_x,pt_ic_y,pt_ic_t,x0=0,y0=0):
    Q_ic_pt = ((1/torch.pi)*torch.exp(-((pt_ic_x)**2+(pt_ic_y)**2)))
    return Variable(Q_ic_pt, requires_grad=False).to(device)

def compute_residual(input_tensors,Q):
    x, y, t = input_tensors[0], input_tensors[1], input_tensors[2]

    Q_x = torch.autograd.grad(Q.sum(), x, create_graph=True)[0]
    Q_y = torch.autograd.grad(Q.sum(), y, create_graph=True)[0]
    Q_t = torch.autograd.grad(Q.sum(), t, create_graph=True)[0]

    Q_xx = torch.autograd.grad(Q_x.sum(), x, create_graph=True)[0]
    Q_yy = torch.autograd.grad(Q_y.sum(), y, create_graph=True)[0]

    residual = gamma*Q + (gamma/2)*x*Q_x + (gamma/2)*y*Q_y + (gamma/4)*(n_bar+1)*(Q_xx+Q_yy) - Q_t

    return residual

def find_ic_indices(input_tensors):
    _, _, pt_t = input_tensors[0], input_tensors[1], input_tensors[2]

    ic_indices = torch.nonzero(pt_t[:,0]==0)
    ic_indices = ic_indices.unique()

    return ic_indices

def find_boundary_indices(input_tensors):
    pt_x, pt_y, pt_t = input_tensors[0], input_tensors[1], input_tensors[2]
    boundary_indices = torch.nonzero(pt_x[:,0]==domain.bounds[0][0])
    boundary_indices = torch.cat((boundary_indices,torch.nonzero(pt_x[:,0]==domain.bounds[0][1])))
    boundary_indices = torch.cat((boundary_indices,torch.nonzero(pt_y[:,0]==domain.bounds[1][0])))
    boundary_indices = torch.cat((boundary_indices,torch.nonzero(pt_y[:,0]==domain.bounds[1][1])))
    boundary_indices = boundary_indices.unique()

    return boundary_indices

def residual_loss(input_tensors,net_output):
    pred_residual_values = compute_residual(input_tensors,net_output)

    true_residual_values = Variable(torch.zeros_like(pred_residual_values).float(), requires_grad=False).to(device)

    return calculate_mse(pred_residual_values,true_residual_values)

def ic_loss(input_tensors,net_output,ic_indices):
    if ic_indices.shape[0] != 0:
        pred_ic_values = net_output[ic_indices]
        pt_x, pt_y, pt_t = input_tensors[0], input_tensors[1], input_tensors[2]
        pt_ic_x, pt_ic_y, pt_ic_t = pt_x[list((ic_indices.cpu().detach().numpy()))], pt_y[list((ic_indices.cpu().detach().numpy()))], pt_t[list((ic_indices.cpu().detach().numpy()))]

        true_ic_values = compute_true_ic_values(pt_ic_x, pt_ic_y, pt_ic_t) 

        return calculate_mse(pred_ic_values, true_ic_values)
    else:
        return 0 

def boundary_loss(input_tensors,net_output,boundary_indices):
        if boundary_indices.shape[0] != 0:
            # compute boundary loss
            pred_boundary_values = net_output[boundary_indices] 
            true_boundary_values = Variable(torch.zeros_like(pred_boundary_values).float(), requires_grad=False).to(device)
            return calculate_mse(pred_boundary_values, true_boundary_values)
        else:
            return 0


# ## 2. Define Hyperparameters for Training

# In[10]:


NUM_LAYERS = 9
NUM_NEURONS_IN_EACH_LAYER = 20 
INITIAlISATION_FUNCTION_NAME = None

X_MIN = -5. 
X_MAX = 5.
Y_MIN = -5.
Y_MAX = 5.
T_MIN = 0.
T_MAX = 1. 

SAMPLER = None
NUM_SAMPLES = 49 

CONVERT_TO_MESHGRID = True 

PATH = 'generate_training_plots'
OPTIMISER = 'adam'
NUM_EPOCHS = 1000

LR_SCHEDULER = False 
LR_SCHEDULER_PATIENCE = max(10,NUM_EPOCHS/10)
LR_SCHEDULER_FACTOR = 0.5


# ## 3. Save the Model's Hyperparameters

# In[11]:


# creating the model hyperparameters dictionary
model_hyperparameters = {
    "NUM_LAYERS": NUM_LAYERS,
    "NUM_NEURONS_IN_EACH_LAYER": NUM_NEURONS_IN_EACH_LAYER,
    "INITIAlISATION_FUNCTION_NAME": INITIAlISATION_FUNCTION_NAME,
    "X_MIN": X_MIN,
    "X_MAX": X_MAX,
    "Y_MIN": Y_MIN,
    "Y_MAX": Y_MAX,
    "T_MIN": T_MIN,
    "T_MAX": T_MAX,
    "SAMPLER": SAMPLER,
    "NUM_SAMPLES": NUM_SAMPLES,
    "CONVERT_TO_MESHGRID": CONVERT_TO_MESHGRID,
    "PATH": PATH,
    "OPTIMISER": OPTIMISER,
    "NUM_EPOCHS": NUM_EPOCHS,
    "LR_SCHEDULER": LR_SCHEDULER,
    "LR_SCHEDULER_PATIENCE": LR_SCHEDULER_PATIENCE,
    "LR_SCHEDULER_FACTOR": LR_SCHEDULER_FACTOR
}

json_file_path = PATH+".json"

with open(json_file_path, 'w') as json_file:
    json.dump(model_hyperparameters, json_file)


# In[12]:


with open(PATH+".json", 'r') as json_file:
    content = json.load(json_file)
    print(json.dumps(content))


# ## 3. Train the Model

# In[13]:


layer_list = [3]+[NUM_NEURONS_IN_EACH_LAYER]*NUM_LAYERS+[1]
network = PINN(layer_list,initialisation_function_name="xavier_normal")
network = network.to(device)


# In[14]:


domain = Domain.from_xy_t(X_MIN, X_MAX, Y_MIN, Y_MAX, T_MIN, T_MAX)
num_samples = NUM_SAMPLES

x_arr, y_arr, t_arr = np.linspace(domain.bounds[0][0], domain.bounds[0][1], num_samples).reshape(num_samples,1), np.linspace(domain.bounds[1][0], domain.bounds[1][1], num_samples).reshape(num_samples,1), np.linspace(domain.bounds[2][0], domain.bounds[2][1], num_samples).reshape(num_samples,1)

x_arr = np.insert(x_arr,0,0)
y_arr = np.insert(y_arr,0,0)
t_arr = np.insert(t_arr,0,0)

if CONVERT_TO_MESHGRID:
    # New flexible meshgrid interface (accepts list of arrays)
    x_arr, y_arr, t_arr = convert_to_meshgrid([x_arr, y_arr, t_arr])
    
    # Alternative: Use backward compatibility function
    # x_arr, y_arr, t_arr = convert_to_meshgrid_xy_t(x_arr, y_arr, t_arr)

loss_list = [{"function":residual_loss, "indices": None, "weight": 1},
             {"function":ic_loss, "indices": find_ic_indices, "weight": 1},
             {"function":boundary_loss, "indices": find_boundary_indices, "weight": 1}]

print("Number of Points: ",x_arr.shape[0])


# In[15]:


125000/10


# In[16]:


network.set_path(PATH)
network.configure_optimiser("adam")

if LR_SCHEDULER:
    network.configure_lr_scheduler("reduce_lr_on_plateau", factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE)
else:
    network.configure_lr_scheduler()

training_history = network.train_model([x_arr, y_arr, t_arr],loss_list,NUM_EPOCHS,125000,monitoring_function=None)

# NOTE: the best loss is calculated on the unbatched data (the validation set) 


# In[11]:


np.save(PATH+'_training_history.npy', training_history)


# In[ ]:


training_history = np.load(PATH+'_training_history.npy')
plot_history(training_history,path=PATH)


# In[ ]:




