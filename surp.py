'''
Successive Refinement Algorithm 
Adoped from SuRP by Isik et al. (2022)
After a model is trained, we apply the SURP algorithm to the model 
We use algorithm 2. 
Hyperparameters: beta 
Input: NN from COIN 
Output: Pruned NN (Tensor) 

Some notes: 
-- We always normalize 
'''
import torch 
import torch.nn as nn
import numpy as np 
from scipy.stats import norm, laplace, geom
import random 
from siren import Siren 
from copy import deepcopy 
import util 
from torchvision import transforms
from torchvision.utils import save_image
import imageio 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Might want to define dtype = torch.float32

class surp:
    def __init__(self, model, beta, total_iter, width, depth, checkpoint_path, image_path):
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
        # Get network weights and related parameters for SuRP
        model, params, param_d, params_abs, signs, norms, lam_inv, checkpoint = self.get_nn_weights(model, checkpoint_path)
        img = imageio.imread(image_path)
        coordinates, features = util.to_coordinates_and_features(img)
        self.model = model 
        self.nn_params = params
        self.param_d = param_d 
        self.params_abs = deepcopy(params_abs) 
        self.params_res = deepcopy(params_abs) # What does res stand for? 
        self.params_abs_recon = torch.zeros_like(params_abs)
        self.signs = signs 
        self.norms = norms
        self.lam_inv = lam_inv
        self.state_dict = checkpoint
        self.n = len(params_abs)
        self.alpha = None # Verify is this beta? 
        self.scale_factor = np.log(float(self.n)/float(np.log(self.n)))
        self.alpha = self.alpha / self.scale_factor
        self.lam_inv = self.alpha * self.lam_inv
        self.gamma = None 
        # self.pruning = pruning
        # Assigning arguments to instance variables
        self.beta = beta
        self.total_iter = total_iter  # Total number of iterations (L in the paper)
        self.width = width  # Network width from argparser
        self.depth = depth  # Network depth from argparser
        self.checkpoint_path = checkpoint_path  # Path to the checkpoint file
        self.img = transforms.ToTensor(img).float().to(device, torch.float32)
        self.coordinates, self.features = coordinates.to(device, torch.float32), features.to(device, torch.float32)

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


    def enc_step(self):
        param_list = self.params_res
        lambda_inv = self.lam_inv
        m_inds = torch.nonzero(param_list.gt(lambda_inv*self.scale_factor))

        if len(m_inds) == 0:
            print('no such index')
            return None, None
        else:
            m = np.random.choice(m_inds.detach().cpu().numpy().reshape(len(m_inds)))
            geom_rv = geom(float(len(m_inds))/float(len(param_list)))  # Declare a geometric random variable
            k = geom_rv.rvs()  # Get a random sample from geom_rv, k to be used for bitrate estimates. TODO: Ask

            # Update \hat{U}:
            self.params_abs_recon[m] = self.params_abs_recon[m] + self.lam_inv*self.scale_factor
            # Update U:
            self.params_res[m] = self.params_res[m] - self.lam_inv*self.scale_factor
            return m, k

    def load_reconst_weights(self, w_hat):
        i = 0
        w_hat = w_hat*self.signs
        w_hat = w_hat*self.norms
        new_state_dict = deepcopy(self.nn_state_d)
        for k, k_shape in self.param_d.items():
            k_size = k_shape.numel()
            new_state_dict[k] = w_hat[i:(i + k_size)].view(k_shape)
            i += k_size
        self.model.load_state_dict(new_state_dict)

    ## TODO: Ask what alpha/beta/gamma are 
    def successive_refine(self):
        refresh_count = 0 # handling the empty list of indices scenario 
        for i in range(self.total_iter):
            m, k = self.enc_step()
            while m is None:
                refresh_count += 1
                if refresh_count % 20 == 0 and refresh_count>1:
                    self.alpha =  self.alpha*0.9
                # Refresh the parameter lambda.
                self.lam_inv = torch.mean(self.params_res)
                self.lam_inv = self.alpha * self.lam_inv
                # Compute m, k again after the parameter lambda is refreshed.
                m, k = self.enc_step()
            
            # for every 100 iterations, reconstruct an image and compute PSNR
            if i % 100 == 0: 

                w_hat = deepcopy(self.params_abs_recon)
                self.load_reconst_weights(w_hat)           
                self.synthesize_image(w_hat)

            # Update Gamma and Lambda 
            self.gamma = (self.n - 1)/(self.n - self.scale_factor)
            self.lam_inv = self.gamma*(self.n - self.scale_factor) / self.n * self.lam_inv

    def synthesize_image(self, w_hat):
        '''
        This will take the SURP network and reconstruct the image. It also computes PSNR
        '''

        # Image Reconstruction 
        with torch.no_grad():
            img_recon = self.model(self.coordinates).reshape(self.img.shape[1], self.img.shape[2], 3).permute(2, 0, 1)
            save_image(torch.clamp(img_recon, 0, 1).to('cpu'), 'test') # define path

        # Compute PSNR 
        psnr = util.get_clamped_psnr(img_recon, self.features) 
        print("PSNR:", psnr)
        
        # Compute sparsity:
        sparsity = torch.sum(w_hat == 0).item()/w_hat.numel()
        print("Sparsity:", sparsity)
    