"""
Apply one of the three compression approaches 
"""

import argparse
import random
import torch
import modules.util as util
from modules.siren import Siren
from torchvision import transforms
from torchvision.utils import save_image
from modules.training import Trainer
from modules.model_compression import pruning, quantization, surp 

parser = argparse.ArgumentParser()
# General 
parser.add_argument("--image_id", help="Image ID to train on, if not the full dataset", type=int)
parser.add_argument("--compression_type", help="Choose one of Mag_Pruning/Quantization/SuRP", type=str)
parser.add_argument("--layer_size", help="Layer sizes as list of ints", type=int) # Depending on trained model size
parser.add_argument("--num_layers", help="Number of layers", type=int) # Depending on trained model size
parser.add_argument("--w0", help="w0 parameter for SIREN model.", type=float, default=30.0)
parser.add_argument("-w0i", "--w0_initial", help="w0 parameter for first layer of SIREN model.", type=float, default=30.0)
# Magnitude Pruning Specific
parser.add_argument("--prune_ratio", help = "Mag_Prune only: pruning ratio", type = float, default = None)
parser.add_argument("--refine_iter", help = "Mag_Prune only: refine iteration", type = int, default = None)
# Quantization Specific
parser.add_argument("--quant_level", help = "Quantization only: quantization level (half only for now)", type = str, default = None)
# SuRP specific 
parser.add_argument("--surp_iter", help="Number of refinement steps", type=int, default=125000)
parser.add_argument("--image_iter", help="Number of steps per image synthesis", type=int, default=2500)
args = parser.parse_args()

# Set up torch and cuda
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

# Set random seeds. Not gonna bother to add an argparser for this one
torch.manual_seed(random.randint(1, int(1e6)))
torch.cuda.manual_seed_all(random.randint(1, int(1e6)))

func_rep = Siren(dim_in=2, dim_hidden=args.layer_size, dim_out=3,
        num_layers=args.num_layers, final_activation=torch.nn.Identity(),
        w0_initial=args.w0_initial, w0=args.w0).to(device)

# Based on compression type, apply corresponding methods
if args.compression_type == "Mag_Pruning":
    compressor = pruning(func_rep, args.image_id, args.compression_type, 
                         args.layer_size, args.num_layers,
                         args.prune_ratio, args.refine_iter)
    compressor.prune()
elif args.compression_type == "Quantization":
    compressor = quantization(func_rep, args.image_id, args.compression_type, 
                         args.layer_size, args.num_layers,
                         args.quant_level)
    compressor.quantize()

elif args.compression_type == "SuRP":
    compressor = surp(func_rep, args.image_id, args.compression_type, 
                    args.layer_size, args.num_layers,
                    args.surp_iter, args.image_iter)
    compressor.successive_refine()
        
