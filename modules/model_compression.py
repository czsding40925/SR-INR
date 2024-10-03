'''
This model compression script contains three post-training compression method
1. Pruning
2. SuRP
3. Quantization

It contains a base class and then three separate classes of those methods. 
'''

import torch 
import torch.nn as nn
import numpy as np 
from scipy.stats import norm, laplace, geom
import random 
from modules.siren import Siren 
from copy import deepcopy 
import modules.util as util 
from torchvision import transforms
from torchvision.utils import save_image
import imageio 
import matplotlib.pyplot as plt
import os
import tqdm
from modules.training import Trainer 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32

class base_model:
    def __init__(self, model, image_id, compression_type, width, depth):
        self.image_id = image_id 
        self.state_dict = torch.load(self.get_model_path)
        self.model = model.load_state_dict(self.state_dict)
        self.compression_type = compression_type
        self.width = width # layer size
        self.depth = depth # lazyer depth 
        self.img = self.load_image()
        coordinates, features = util.to_coordinates_and_features(self.img)
        self.coordinates, self.features = coordinates.to(device, dtype), features.to(device, dtype)
        self.image_save_path = self.get_save_path()
        
    def load_image(self):
        img = imageio.imread(f"kodak-dataset/kodim{str(self.image_id).zfill(2)}.png")
        img = transforms.ToTensor()(img).float().to(device, dtype)
        return img 
    
    def get_model_path(self):
        model_path = f"results/image_{self.image_id}/best_model_{self.image_id}"
        return model_path

    def get_save_path(self):
        if self.compression_type != 'SuRP':
            path = f"results/image_{self.image_id}"
        else:
            path = f"results/image_{self.image_id}/SuRP"
        if not os.path.exists(path):
            os.makedirs(path)
        return path 
    
    def synthesize_image(self, iter = None):
        # type full_precision/sr/quantized/pruned
        with torch.no_grad:
            img_recon = self.model(self.coordinates).reshape(self.img.shape[1], self.img.shape[2], 3).permute(2, 0, 1)
            if iter is None: 
                save_image(torch.clamp(img_recon, 0, 1).to('cpu'), self.image_save_path + f'/{self.compression_type}_reconstruction_{self.image_id}.png')
            else: 
                # to handle successive refinement iteration case 
                save_image(torch.clamp(img_recon, 0, 1).to('cpu'), self.image_save_path + f'/{self.compression_type}_reconstruction_{self.image_id}_{iter}.png')
            print(f"Image Saved at {self.image_save_path}/{type}_reconstruction_{self.image_id}.png" )
        psnr = util.get_clamped_psnr(self.img, img_recon)
        return img_recon, psnr  
    
    def save_model(self):
        path = self.image_save_path+f'{self.compression_type}_model_{self.image_id}'
        torch.save(self.model, path)
        print(f"Model saved at {path}")


class pruning(base_model):
    def __init__(self, model, image_id, compression_type, width, depth, pruning_ratio, refine_iter):
        super.__init__(model, image_id, compression_type, width, depth)
        self.pruning_percent = pruning_ratio
        self.refine_iter = refine_iter
    
    def prune(self):
        self.model, masks = util.apply_magnitude_pruning(self.model, pruning_percent=self.pruning_ratio)
        trainer = Trainer(self.model, lr=1e-3, sparse_training=True, masks=masks)
        trainer.train(self.coordinates, self.features, num_iters=self.refine_iter)
        self.save_model()
        psnr = self.synthesize_image()
        print("PSNR:", psnr)


class quantization(base_model):
    def __init__(self, model, image_id, compression_type, width, depth, quantization_mode):
        super.__init__(model, image_id, compression_type, width, depth)
        self.quantization_mode = quantization_mode

    def quantize(self):
        if self.quantization_mode == 0.5:
            self.model.half().to(device)
        self.save_model()
        psnr = self.synthesize_image()
        print("PSNR:", psnr)


class surp(base_model):
    def __init__(self, model, image_id, compression_type, width, depth, total_iter, img_iter):
        super.__init__(model, image_id, compression_type, width, depth)
        self.params, self.param_d, params_abs, self.signs, self.norms, self.lam_inv = self.get_nn_weights()
        self.params_abs = deepcopy(params_abs) 
        self.params_res = deepcopy(params_abs) # What does res stand for? 
        self.params_abs_recon = torch.zeros_like(params_abs)
        self.total_iter = total_iter # Total number of iterations (L in the paper)
        self.img_iter = img_iter # Generate image per 
        self.n = len(self.param_abs)
        self.scale_factor = np.log(float(self.n)/float(np.log(self.n)))
        self.alpha = 10 / self.scale_factor # might change 10 to others
        self.lam_inv = self.alpha * self.lam_inv
        self.gamma = 1 # From the config files (resnet/vgg). What is this? 


    def get_nn_weights(self):
        param_d = {}
        with torch.no_grad():
            # Load the checkpoint
            # checkpoint = torch.load(checkpoint_path)
            # model.load_state_dict(checkpoint)

            # Prepare to collect model parameters
            # self.model.eval()
            params = []
            norms = []
            print('Target network weights:')
            
            for (name, p) in self.model.named_parameters():
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
            params, norms = torch.cat(params), torch.cat(norms)

            # Save sign and absolute values of weights
            signs = torch.sign(params).float().cuda()  # Convert to float and move to GPU
            params_abs = torch.abs(params)
            
            # Compute the mean of weight magnitudes
            lam_inv = torch.mean(params_abs)
            print(f'Mean of the magnitudes is: {lam_inv}')

        print(f'Total target network params: {len(params)}\n')

        return params, param_d, params_abs, signs, norms, lam_inv
    
    def enc_step(self):
        param_list = self.params_res
        lambda_inv = self.lam_inv
        m_inds = torch.nonzero(param_list.gt(lambda_inv*self.scale_factor))

        if len(m_inds) == 0:
            # print('no such index')
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
        new_state_dict = deepcopy(self.nn_state_dict)
        for k, k_shape in self.param_d.items():
            k_size = k_shape.numel()
            new_state_dict[k] = w_hat[i:(i + k_size)].view(k_shape)
            i += k_size
        self.model.load_state_dict(new_state_dict)

    def successive_refine(self):
        refresh_count = 0  # Handling the empty list of indices scenario

        iters = []
        spars = []
        psnrs = []
        
        # Create a tqdm progress bar
        with tqdm.trange(self.total_iter, ncols=100) as t:
            for i in t:
                m, k = self.enc_step()
                while m is None:
                    refresh_count += 1
                    if refresh_count % 20 == 0 and refresh_count > 1:
                        self.alpha = self.alpha * 0.9
                    
                    # Refresh the parameter lambda.
                    self.lam_inv = torch.mean(self.params_res)
                    self.lam_inv = self.alpha * self.lam_inv
                    
                    # Compute m, k again after the parameter lambda is refreshed.
                    m, k = self.enc_step()

                # For every 5000 iterations, reconstruct an image and compute PSNR
                if i % self.img_iter == 0:
                    #print("Current Iteration:", i)
                    w_hat = deepcopy(self.params_abs_recon)
                    self.load_reconst_weights(w_hat)
                    psnr = self.synthesize_image(type = 'SuRP', iter = i)
                    sparsity = self.compute_sparsity(w_hat)
                    # Log iteration, PSNR, and sparsity
                    iters.append(i)
                    psnrs.append(psnr)
                    spars.append(sparsity)
                    
                # Update tqdm bar with the latest PSNR and sparsity
                t.set_postfix(psnr=f"{psnr:.2f}", sparsity=f"{sparsity:.2f}")

                # Update Gamma and Lambda
                self.gamma = (self.n - 1) / (self.n - self.scale_factor)
                self.lam_inv = self.gamma * (self.n - self.scale_factor) / self.n * self.lam_inv

        # After the loop, save the plot
        self.plot_psnr_sparsity(iters, spars, psnrs)
        self.save_model()
    
    def compute_sparsity(self, w_hat):
        return torch.sum(w_hat == 0).item()/w_hat.numel()

    def plot_psnr_sparsity(self, iters, spars, psnrs):
        """
        Creates a single plot with two y-axes: 
        - Iterations vs. Sparsity on the left y-axis
        - Iterations vs. PSNR on the right y-axis

        Args:
            iters (np.ndarray): Array of iteration values.
            spars (np.ndarray): Array of sparsity values corresponding to the iterations.
            psnrs (np.ndarray): Array of PSNR values corresponding to the iterations.
        """
        # Create a figure and a single set of axes
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot Iterations vs. Sparsity on the left y-axis
        ax1.plot(iters, spars, marker='o', linestyle='-', color='b', label='Sparsity')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Sparsity', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True)
        
        # Create a second y-axis sharing the same x-axis for PSNR
        ax2 = ax1.twinx()  
        ax2.plot(iters, psnrs, marker='o', linestyle='-', color='r', label='PSNR')
        ax2.set_ylabel('PSNR', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # Set title for the plot
        plt.title('Iterations vs. Sparsity and PSNR')

        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

        # Adjust layout for better spacing
        fig.tight_layout()

        # Save the plot to the specified path
        image_save_path = "results/sr_images"
        os.makedirs(image_save_path, exist_ok=True)
        plt.savefig(os.path.join(self.image_save_path, "sparsity_psnr_plot.png"))
        print(f"Plot saved at {os.path.join(self.image_save_path, 'sparsity_psnr_plot.png')}")
        # plt.show()
