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
from siren import Siren 
from copy import deepcopy
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class surp:
    def __init__(self, model, beta, total_iter, width, depth, checkpoint_path):
        """
        Applying the SuRP algorithm to a given Neural Network (NN).

        Args:
            model: The trained neural network model to be refined.
            beta (float): SuRP hyperparameter controlling sparsity.
            total_iter (int): Total number of iterations for the SuRP refinement.
            width (int): Width of the network (from argparser).
            depth (int): Depth of the network (from argparser).
            checkpoint_path (str): Path to the checkpoint file containing the model weights.
        """
        # Get network weights and related parameters
        self.model, self.params, self.param_d, self.params_abs, self.signs, self.norms, self.lam_inv, self.checkpoint = self.get_nn_weights(model, checkpoint_path)
        
        # Assigning arguments to instance variables
        self.beta = beta
        self.total_iter = total_iter  # Total number of iterations (L in the paper)
        self.width = width  # Network width from argparser
        self.depth = depth  # Network depth from argparser
        self.checkpoint_path = checkpoint_path  # Path to the checkpoint file

    # Implement the get_nn_weights method
    def get_nn_weights(self, model, checkpoint_path):
        """
        Retrieves and processes the weights of the given neural network model.

        Args:
            model: The neural network model.
            checkpoint_path (str): Path to the checkpoint file containing the model weights.

        Returns:
            tuple: Contains the model, parameters, parameter dictionary, 
                   absolute parameters, signs, norms, lambda inverse, and checkpoint.
        """
        param_d = {}
        with torch.no_grad():
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint)

            # Prepare to collect model parameters
            model.eval()
            params = []
            norms = []
            print('Target network weights:')
            
            for (name, p) in model.named_parameters():
                if p.requires_grad:
                    weights = deepcopy(p.view(-1))
                    norm = torch.norm(weights)
                    norms.append(norm * torch.ones_like(weights))
                    weights = weights / norm  # Normalize the weights
                    
                    # Collect parameters and their dimensions
                    params.append(weights)
                    param_d[name] = p.size()
                    print(f'{name}: {p.size()}')

            # Concatenate all the parameters and norms
            params = torch.cat(params)
            norms = torch.cat(norms)

            # Save sign and absolute values of weights
            signs = torch.sign(params).float().cuda()  # Convert to float and move to GPU
            params_abs = torch.abs(params)
            
            # Compute the mean of weight magnitudes
            lam_inv = torch.mean(params_abs)
            print(f'Mean of the magnitudes is: {lam_inv}')

        print(f'Total target network params: {len(params)}\n')

        return model, params, param_d, params_abs, signs, norms, lam_inv, checkpoint




    # def l1_normalize(self, weights):
    #     """
    #     Perform L1 normalization on the given weights.
    #     """
    #     l1_norm = torch.norm(weights, p=1, dim=1, keepdim=True)
    #     normalized_weights = weights / (l1_norm + 1e-6)  # Adding a small value to prevent division by zero
    #     return normalized_weights

    # def apply_l1_normalization_to_model(self):
    #     """
    #     Apply L1 normalization to the weights of all linear layers in the model.
    #     """
    #     for name, param in self.model.named_parameters():
    #         if 'weight' in name:
    #             with torch.no_grad():
    #                 # Normalize the weights of each layer using L1 normalization
    #                 normalized_weight = self.l1_normalize(param.data)
    #                 param.data.copy_(normalized_weight)

    # def concat_weights(self):
    #     # Concatenate weights 
    #     all_weights = [] 
    #     for name, param in self.model.named_parameters():
    #         if 'weight' in name and param.requires_grad:
    #             weights = param.detach().cpu().numpy()  # Get the weights as a NumPy array
    #             all_weights.append(weights.flatten())   # Flatten the weights and store them

    #     all_weights = np.concatenate(all_weights)
    #     return all_weights 
    
    
    # # Algorithm 1 of the SuRP paper 
    # def successive_refine(self):
    #     # Normalization by Layer 
    #     self.apply_l1_normalization_to_model()
        
    #     # Get all weights as a list 
    #     all_weights = self.concat_weights()

    #     # Initialize the U vector for successive refinement
    #     n = len(all_weights)
    #     U = np.copy(all_weights)
    #     U_recon = np.zeros(n)

    #     # First estimation of Lambda (mean = 1/ Lambda)
    #     lambda_laplace = 1/np.mean(np.abs(all_weights))

    #     # Iteratively refine
    #     for i in range(self.total_iter):
    #         thresh = (1 / lambda_laplace) * np.log(n/2*self.beta)

    #         # Find indices 
    #         m_max = np.where(U > thresh)
    #         m_min = np.where(U < -thresh)
    #         # Handle the empty case 
    #         if len(m_max) == 0 or len(m_min)==0:
    #             lambda_laplace = 1/np.mean(all_weights)
    #             # continue 
    #         m_plus = random.choice(m_max)
    #         m_minus = random.choice(m_min)
    #         U[m_plus] -= (1 / lambda_laplace) * np.log(n/2*self.beta)
    #         U[m_minus] += (1 / lambda_laplace) * np.log(n/2*self.beta)
    #         lambda_laplace *= n/(n-2 * np.log(n/self.beta)) 

    #         # Decoder 
    #         U_recon[m_plus] += thresh 
    #         U_recon[m_minus] -= thresh 

    #     # Return 
    #     # Denormalize the weights 
    #     # all_weights_refined = U * np.sum(np.abs(all_weights))

    #     """
    #     # Encoder sends lambda_laplace to the Decoder 
    #     for i in range(L):
    #         thresh = (1 / lambda_laplace) * np.log(n/beta)

    #         # Find indices 
    #         m_inds = np.where(all_weights > thresh)
    #         # Handle the empty case 
    #         if len(m_inds) > 0:
    #             m = random.choice(m_inds)
    #             all_weights[m] -= thresh 
    #             lambda_laplace *= n/(n-np.log(n/beta)) 
    #             U[m] += thresh 
    #         else: 
    #             lambda_laplace *= n/(n-np.log(n/beta)) 

    #     # Denormalize the weights 
    #     # TODO: how to denormalize in this case? 
    #     # Convert the sparse array back to an NN
    #     # TODO: Ask Berivan how the network weights is converted 
    #     pruned_NN = None 
    #     return pruned_NN
    
    # """