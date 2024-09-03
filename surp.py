'''
Successive Refinement Algorithm 
Adoped from SuRP by Isik et al. (2022)
After a model is trained, we apply the SURP algorithm to the model 
We use algorithm 2. 
Hyperparameters: beta 
Input: NN from COIN 
Output: Pruned NN (Tensor) 
'''
import torch 
import torch.nn as nn
import numpy as np 
from scipy.stats import norm, laplace
import random 

def l1_normalize(weights):
    """
    Perform L1 normalization on the given weights.
    """
    l1_norm = torch.norm(weights, p=1, dim=1, keepdim=True)
    normalized_weights = weights / (l1_norm + 1e-6)  # Adding a small value to prevent division by zero
    return normalized_weights

def apply_l1_normalization_to_model(model):
    """
    Apply L1 normalization to the weights of all linear layers in the model.
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            with torch.no_grad():
                normalized_weight = l1_normalize(param.data)
                param.data.copy_(normalized_weight)

# TODO: Change ReLU to SIREN activation  
def array_to_fully_connected_nn(array, input_dim, layer_width, output_dim, depth):
    # Calculate the total number of parameters needed for each layer
    current_index = 0
    layers = []
    
    # Input to first hidden layer
    weight_shape = (layer_width, input_dim)
    weight_size = weight_shape[0] * weight_shape[1]
    weights = array[current_index:current_index + weight_size].reshape(weight_shape)
    current_index += weight_size
    
    bias_shape = (layer_width,)
    bias_size = bias_shape[0]
    biases = array[current_index:current_index + bias_size]
    current_index += bias_size
    
    fc1 = nn.Linear(input_dim, layer_width)
    with torch.no_grad():
        fc1.weight.copy_(torch.tensor(weights))
        fc1.bias.copy_(torch.tensor(biases))
    layers.append(fc1)
    layers.append(nn.ReLU())  # Assuming ReLU activation

    # Hidden layers
    for _ in range(depth - 1):
        weight_shape = (layer_width, layer_width)
        weight_size = weight_shape[0] * weight_shape[1]
        weights = array[current_index:current_index + weight_size].reshape(weight_shape)
        current_index += weight_size
        
        bias_shape = (layer_width,)
        bias_size = bias_shape[0]
        biases = array[current_index:current_index + bias_size]
        current_index += bias_size
        
        fc = nn.Linear(layer_width, layer_width)
        with torch.no_grad():
            fc.weight.copy_(torch.tensor(weights))
            fc.bias.copy_(torch.tensor(biases))
        layers.append(fc)
        layers.append(nn.ReLU())

    # Last hidden layer to output
    weight_shape = (output_dim, layer_width)
    weight_size = weight_shape[0] * weight_shape[1]
    weights = array[current_index:current_index + weight_size].reshape(weight_shape)
    current_index += weight_size
    
    bias_shape = (output_dim,)
    bias_size = bias_shape[0]
    biases = array[current_index:current_index + bias_size]
    current_index += bias_size
    
    fc_output = nn.Linear(layer_width, output_dim)
    with torch.no_grad():
        fc_output.weight.copy_(torch.tensor(weights))
        fc_output.bias.copy_(torch.tensor(biases))
    layers.append(fc_output)

    # Create the final model
    model = nn.Sequential(*layers)
    return model


# TODO: ask/find out what beta is 
def SURP(NN, depth, width, beta = 8, L = 10):
    # Normalization by Layer 
    apply_l1_normalization_to_model(NN)
    
    # Convert the Normalized NN to an array (all_weights)
    all_weights = []
    for name, param in NN.named_parameters():
        if 'weight' in name and param.requires_grad:
          weights = param.detach().cpu().numpy()  # Get the weights as a NumPy array
          all_weights.append(weights.flatten())   # Flatten the weights and store them

    all_weights = np.concatenate(all_weights)

    # Initialize the U vector for successive refinement
    n = len(all_weights)
    U = np.zeros(n)

    # Estimate the lambda value of Laplace distribution for all_weights 
    # Using scipy fits here (verify)
    mu_laplace, lambda_laplace = laplace.fit(all_weights)

    # Encoder sends lambda_laplace to the Decoder 
    for i in range(L):
        thresh = (1 / lambda_laplace) * np.log(n/beta)

        # Find indices 
        m_inds = np.where(all_weights > thresh)
        # Handle the empty case 
        if len(m_inds) > 0:
            m = random.choice(m_inds)
            all_weights[m] -= thresh 
            lambda_laplace *= n/(n-np.log(n/beta)) 
            U[m] += thresh 
        else: 
            lambda_laplace *= n/(n-np.log(n/beta)) 

    # Denormalize the weights 
    # TODO: how to denormalize in this case? 
    # Convert the sparse array back to an NN
    input_dim = None
    output_dim = None 
    pruned_NN = array_to_fully_connected_nn(U, input_dim, width, output_dim, depth)
    return pruned_NN
