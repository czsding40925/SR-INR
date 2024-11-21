import numpy as np
import torch
from torch._C import dtype
from typing import Dict
import copy 
import os
import imageio 
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from copy import deepcopy 
from torchvision import transforms
from torchvision.utils import save_image

DTYPE_BIT_SIZE: Dict[dtype, int] = {
    torch.float32: 32,
    torch.float: 32,
    torch.float64: 64,
    torch.double: 64,
    torch.float16: 16,
    torch.half: 16,
    torch.bfloat16: 16,
    torch.complex32: 32,
    torch.complex64: 64,
    torch.complex128: 128,
    torch.cdouble: 128,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.short: 16,
    torch.int32: 32,
    torch.int: 32,
    torch.int64: 64,
    torch.long: 64,
    torch.bool: 1
}


def svd_analysis(image_id: int):
    """
    Looks at the singular values of each layer in the model and shares the number of 
    singular values to explain 50%, 75%, 90%, and 95% of the variance in the data.

    Args:
        image_id (int): ID corresponding to image on which model was fit.
    """
    state_dict = torch.load(f"results/image_{image_id}/best_model_{image_id}.pt")    
    print(f"[IMAGE ID {image_id}]")
    for (name, p) in state_dict.items():
        if "weight" in name:
            u, s, v = torch.linalg.svd(p)
            singular_values_squared = s**2
            total_variance = singular_values_squared.sum()
            explained_variance_ratio = singular_values_squared / total_variance
            variance_accumulator = 0
            variance_notification = [1, 1, 1, 1]
            for i in range(len(explained_variance_ratio)):
                variance_accumulator += explained_variance_ratio[i]
                if variance_accumulator >= 0.95 and variance_notification[0]:
                    variance_notification = [0,0,0,0]
                    print(f"{i} principal components to explain 95% variance for {name}")
                elif variance_accumulator >= 0.90 and variance_notification[1]:
                    variance_notification = [1,0,0,0]
                    print(f"{i} principal components to explain 90% variance for {name}")
                elif variance_accumulator >= 0.75 and variance_notification[2]:
                    variance_notification = [1,1,0,0]
                    print(f"{i} principal components to explain 75% variance for {name}")
                elif variance_accumulator >= 0.5 and variance_notification[3]:
                    variance_notification = [1,1,1,0]
                    print(f"{i} principal components to explain 50% variance for {name}")          
            print()


def to_coordinates_and_features(img):
    """Converts an image to a set of coordinates and features.

    Args:
        img (torch.Tensor): Shape (channels, height, width).
    """
    # Coordinates are indices of all non zero locations of a tensor of ones of
    # same shape as spatial dimensions of image
    coordinates = torch.ones(img.shape[1:]).nonzero(as_tuple=False).float()
    # Normalize coordinates to lie in [-.5, .5]
    coordinates = coordinates / (img.shape[1] - 1) - 0.5
    # Convert to range [-1, 1]
    coordinates *= 2
    # Convert image to a tensor of features of shape (num_points, channels)
    features = img.reshape(img.shape[0], -1).T
    return coordinates, features


def model_size_in_bits(model):
    """Calculate total number of bits to store `model` parameters and buffers."""
    return sum(sum(t.nelement() * DTYPE_BIT_SIZE[t.dtype] for t in tensors)
               for tensors in (model.parameters(), model.buffers()))


def bpp(image, model):
    """Computes size in bits per pixel of model.

    Args:
        image (torch.Tensor): Image to be fitted by model.
        model (torch.nn.Module): Model used to fit image.
    """
    num_pixels = np.prod(image.shape) / 3  # Dividing by 3 because of RGB channels
    return model_size_in_bits(model=model) / num_pixels


def psnr(img1, img2):
    """Calculates PSNR between two images.

    Args:
        img1 (torch.Tensor):
        img2 (torch.Tensor):
    """
    return 20. * np.log10(1.) - 10. * (img1 - img2).detach().pow(2).mean().log10().to('cpu').item()


def clamp_image(img):
    """Clamp image values to like in [0, 1] and convert to unsigned int.

    Args:
        img (torch.Tensor):
    """
    # Values may lie outside [0, 1], so clamp input
    img_ = torch.clamp(img, 0., 1.)
    # Pixel values lie in {0, ..., 255}, so round float tensor
    return torch.round(img_ * 255) / 255.


def get_clamped_psnr(img, img_recon):
    """Get PSNR between true image and reconstructed image. As reconstructed
    image comes from output of neural net, ensure that values like in [0, 1] and
    are unsigned ints.

    Args:
        img (torch.Tensor): Ground truth image.
        img_recon (torch.Tensor): Image reconstructed by model.
    """
    return psnr(img, clamp_image(img_recon))


def mean(list_):
    return np.mean(list_)

def apply_magnitude_pruning(trained_model, pruning_percent=0.2):
    """
    Prunes the weights of the trained_model by setting the smallest weights to zero and returns a new pruned model.
    
    Args:
        trained_model: The pre-trained PyTorch model to prune.
        pruning_percent: The percentage of weights to prune (e.g., 0.2 for 20%).

    Returns:
        pruned_model: A copy of the trained_model with pruned weights.
        masks: A dictionary of masks indicating which weights are pruned (0) and which are trainable (1).
    """
    # Create a deep copy of the trained model to apply pruning
    pruned_model = copy.deepcopy(trained_model)

    # Create a list to store all weights across layers and their corresponding masks
    masks = {}

    # Collect all weights across layers
    all_weights = []

    for name, param in pruned_model.named_parameters():
        if 'weight' in name:
            all_weights.append(param.data.abs().clone().view(-1))  # Flatten the weights to apply pruning

    # Concatenate all weights into a single tensor
    all_weights = torch.cat(all_weights)

    # Determine the threshold for pruning (the smallest values to zero)
    threshold = torch.quantile(all_weights, pruning_percent)

    # Prune each layer by zeroing out weights below the threshold and create a mask
    for name, param in pruned_model.named_parameters():
        if 'weight' in name:
            # Create a mask for the weights: 1 for unpruned weights, 0 for pruned weights
            mask = (param.data.abs() > threshold).float()
            masks[name] = mask  # Save the mask

            # Zero out the pruned weights
            param.data.mul_(mask)

    return pruned_model, masks

def extract_weights(model):
  # Extract Weights
    all_weights = []
    for name, param in model.named_parameters():
      if 'weight' in name and param.requires_grad:
          weights = param.detach().cpu().numpy()  # Get the weights as a NumPy array
          all_weights.append(weights.flatten())   # Flatten the weights and store them

    all_weights = np.concatenate(all_weights)

    return all_weights

def compute_ms_ssim(image_1, image_2):
    image2 = torch.clamp(image_2, 0, 1)  # Clamps values between 0 and 1
    return ms_ssim(image_1.unsqueeze(0), image_2.unsqueeze(0), data_range=1, size_average=True).item()



