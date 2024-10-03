"""Train a new INR model"""

import argparse
import imageio.v2 as imageio
import json
import os
import random
import torch
import modules.util as util
from modules.siren import Siren
from torchvision import transforms
from torchvision.utils import save_image
from modules.training import Trainer
# Set up torch and cuda
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

# Set random seeds. Not gonna bother to add an argparser for this one
torch.manual_seed(random.randint(1, int(1e6)))
torch.cuda.manual_seed_all(random.randint(1, int(1e6)))

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", help="Path to save logs", default="results")
parser.add_argument("--num_iters", help="Number of iterations to train for", type=int, default=50000)
parser.add_argument("--learning_rate", help="Learning rate", type=float, default=2e-4)
parser.add_argument("--image_id", help="Image ID to train on, if not the full dataset", type=int, default=15)
parser.add_argument("--layer_size", help="Layer sizes as list of ints", type=int, default=28)
parser.add_argument("--num_layers", help="Number of layers", type=int, default=10)
parser.add_argument("--w0", help="w0 parameter for SIREN model.", type=float, default=30.0)
parser.add_argument("--w0_initial", help="w0 parameter for first layer of SIREN model.", type=float, default=30.0)
args = parser.parse_args()

# Create directory to store experiments
path = args.logdir + f'/image_{args.image_id}/'
if not os.path.exists(path):
    os.makedirs(path)

# Load image
img = imageio.imread(f"kodak-dataset/kodim{str(args.image_id).zfill(2)}.png")
img = transforms.ToTensor()(img).float().to(device, dtype)

# Setup model
func_rep = Siren(dim_in=2, dim_hidden=args.layer_size, dim_out=3,
    num_layers=args.num_layers, final_activation=torch.nn.Identity(),
    w0_initial=args.w0_initial, w0=args.w0).to(device)

print(f"Fitting SIREN to Kodak Image{args.image_id}...")
# Set up training
trainer = Trainer(func_rep, lr=args.learning_rate)
coordinates, features = util.to_coordinates_and_features(img)
coordinates, features = coordinates.to(device, dtype), features.to(device, dtype)

# Calculate model size. Divide by 8000 to go from bits to kB
model_size = util.model_size_in_bits(func_rep) / 8000.
print(f'Model size: {model_size:.1f}kB')
fp_bpp = util.bpp(model=func_rep, image=img)
print(f'Full precision bpp: {fp_bpp:.2f}')

# Train model in full precision
trainer.train(coordinates, features, num_iters=args.num_iters)
print(f'Best training psnr: {trainer.best_vals["psnr"]:.2f}')
# Save best model
torch.save(trainer.best_model, os.path.join(path, f'best_model_{args.image_id}'))