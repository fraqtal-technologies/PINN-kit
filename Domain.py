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
    def __init__(self,x_min,x_max,y_min,y_max,t_min,t_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min 
        self.y_max = y_max
        self.t_min = t_min
        self.t_max = t_max

    def sample_points(self,num_samples,sampler=None,fixed_time=None):

        # generates evenly spaced samples
        if sampler == None:
            x_arr = np.linspace(self.x_min,self.x_max,num_samples).reshape(num_samples,1)
            y_arr = np.linspace(self.y_min,self.y_max,num_samples).reshape(num_samples,1)
            if fixed_time == None:
                # fixed_time is none (generate evenly spaced points in time)
                t_arr = np.linspace(self.t_min,self.t_max,num_samples).reshape(num_samples,1)
            else:
                # fixed_time is some float (generate points of a fixed value fixed_time)
                t_arr = np.ones([num_samples,1])*fixed_time

        elif sampler == "random":
            x_arr = np.random.uniform(low=self.x_min, high=self.x_max, size=(num_samples,1))
            y_arr = np.random.uniform(low=self.y_min, high=self.y_max, size=(num_samples,1))
            if fixed_time == None:
                # fixed_time is none (generate evenly spaced points in time)
                t_arr = np.random.uniform(low=self.t_min, high=self.t_max, size=(num_samples,1))
            else:
                # fixed_time is some float (generate points of a fixed value fixed_time)
                t_arr = np.ones([num_samples,1])*fixed_time

        else:
            skip = 0
            # Latin Hypercube Sampling
            if sampler == "lhs_classic":
                sampler = Lhs(lhs_type="classic")
            elif sampler == "lhs_centered":
                sampler = Lhs(lhs_type="centered")
            elif sampler == "halton":
                skip = 1
                sampler = Halton()
            elif sampler == "hammersly":
                skip = 1
                sampler = Hammersly()
            elif sampler == "sobol":
                skip = 1
                sampler = Sobol(randomize=False)
            else:
                print("This method is not available.")
                return None,None,None

            if fixed_time == None:
                space = Space([(self.x_min,self.x_max),(self.y_min,self.y_max),(self.t_min,self.t_max)])

                # X will have shape [num_samples,num_dimensions]    
                X = np.asarray(sampler.generate(space.dimensions, num_samples + skip, random_state=10)[skip:])
                x_arr = X[:,0].reshape(num_samples,1)
                y_arr = X[:,1].reshape(num_samples,1)

                # fixed_time is none (generate evenly spaced points in time)
                t_arr = X[:,2].reshape(num_samples,1)
            else:
                space = Space([(self.x_min,self.x_max),(self.y_min,self.y_max)])

                # X will have shape [num_samples,num_dimensions]    
                X = np.asarray(sampler.generate(space.dimensions, num_samples, random_state=10))
                x_arr = X[:,0].reshape(num_samples,1)
                y_arr = X[:,1].reshape(num_samples,1)
                
                # fixed_time is some float (generate points of a fixed value fixed_time)
                t_arr = np.ones([num_samples,1])*fixed_time

        return x_arr, y_arr, t_arr
    
    def sample_bc_points(self,x0,y0,num_samples,sampler=None):
        x_arr, y_arr, t_arr = self.sample_points(num_samples-1,sampler=sampler,fixed_time=0.)
        x_arr = np.insert(x_arr,0,x0,axis=0)
        y_arr = np.insert(y_arr,0,y0,axis=0)
        t_arr = np.insert(t_arr,0,0.,axis=0)
        return x_arr, y_arr, t_arr 
    
# MODIFIED
def convert_to_torch_tensors(input_arrays):
    input_tensors = []
    for input_array in input_arrays:
        input_tensors.append(torch.from_numpy(input_array).float())
    return input_tensors

def to_variable(domain):

    if isinstance(domain,list):
        domain_list_tensor=[]
        for i in domain:
            domain_list_tensor.append(Variable(torch.from_numpy(i.reshape(len(i),1)).float(),requires_grad=True).to(device))
        return domain_list_tensor
    if isinstance(domain,np.ndarray):
        return Variable(torch.from_numpy(domain).float(),requires_grad=True).to(device)
    

def convert_to_meshgrid(x_arr,y_arr,t_arr):
    num_samples = x_arr.shape[0]*y_arr.shape[0]*t_arr.shape[0]

    xx, yy, tt = np.meshgrid(x_arr,y_arr,t_arr)

    return xx.reshape(num_samples,1), yy.reshape(num_samples,1), tt.reshape(num_samples,1)

def plot_points(x_arr,y_arr,title="Meshgrid of Points"):
    # plot meshgrid points
    plt.scatter(x_arr, y_arr, color='blue', s=5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_time_points(t_arr):
    plot_points(t_arr,np.ones(t_arr.shape),title="Time Values")

# MODIFIED 

# extend pytorch dataset class so that we can use dataloader method
class DomainDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        # returns the number of rows in data (the number of samples)
        return len(self.data)

    def __getitem__(self, idx):
        # returns the idx-th row of the data
        return self.data[idx]
    
def gen_batch_data(pt_domain,batch_size=10):
    data = np.hstack(tuple(pt_domain))
    dataset = DomainDataset(data)
    dataloader_list = list(DataLoader(dataset, batch_size=batch_size, shuffle=True))
    
    for i in range(len(dataloader_list)):
        dataloader_list[i] = dataloader_list[i].float()

    return dataloader_list