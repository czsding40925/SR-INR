"""
Apply SuRP 
"""

import argparse
import getpass
import imageio
import json
import os
import random
import torch
import util
from surp import surp
from siren import Siren
from torchvision import transforms
from torchvision.utils import save_image
from training import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("-ld", "--logdir", help="Path to save logs", default="./results/best_model_9.pt")
parser.add_argument("-lss", "--layer_size", help="Layer sizes as list of ints", type=int, default=28)
parser.add_argument("-nl", "--num_layers", help="Number of layers", type=int, default=10)
parser.add_argument("-w0", "--w0", help="w0 parameter for SIREN model.", type=float, default=30.0)
parser.add_argument("-w0i", "--w0_initial", help="w0 parameter for first layer of SIREN model.", type=float, default=30.0)
parser.add_argument("-iid", "--image_id", help="Image ID to train on, if not the full dataset", type=int, default=15)
args = parser.parse_args()

# Set up torch and cuda
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')
image_path = f"kodak-dataset/kodim{str(args.image_id).zfill(2)}.png"

func_rep = Siren(
        dim_in=2,
        dim_hidden=args.layer_size,
        dim_out=3,
        num_layers=args.num_layers,
        final_activation=torch.nn.Identity(),
        w0_initial=args.w0_initial,
        w0=args.w0
    ).to(device)

SR_INR = surp(func_rep, beta = 1, 
            total_iter=1000, 
            width = args.layer_size, 
            depth = args.num_layers, 
            checkpoint_path = args.logdir,
            image_path= image_path)

SR_INR.successive_refine()
#model, params, param_d, params_abs, signs, norms, lam_inv, checkpoint = SR_INR.get_nn_weights(func_rep, args.logdir)


# Run this once should lead to a sparse NN, a reconstructed image, a computed PSNR 