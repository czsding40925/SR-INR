'''
1. We start by training a single layer network 
2. Sparsify using SuRP (Hold for now)
3. Freeze the layer and add a new layer
4. Train and repeat. 
'''

import argparse
import imageio.v2 as imageio
import json
import os
import random
import torch
import torch.nn as nn
import modules.util as util
from modules.siren import Siren, SirenLayer
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

class ExtendedSirenWithInsertion(nn.Module):
    '''
    Calling this will insert a trainable layer right before the final layer 
    '''
    def __init__(self, original_siren, dim_hidden):
        super(ExtendedSirenWithInsertion, self).__init__()
        # Extract the layers up to the second-to-last layer
        original_layers = list(original_siren.net)

        # Insert the new trainable layer just before the last layer
        self.net = nn.Sequential(
            *original_layers,
            SirenLayer(dim_in=dim_hidden, dim_out=dim_hidden, w0=30, use_bias=True)
        )

        # Freeze the parameters of the original layers
        for param in self.net[:-1].parameters():  # Exclude the newly added layer
            param.requires_grad = False

        # Reassign the original last layer as the final layer
        self.last_layer = original_siren.last_layer

    def forward(self, x):
        # Forward through modified network with inserted layer
        x = self.net(x)
        return self.last_layer(x)
    
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
    num_layers=1, final_activation=torch.nn.Identity(), # 1 
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
torch.save(trainer.best_model, os.path.join(path, f'With_No_Grad_{args.image_id}_layer_1.pt'))

# Reconstruct images
# Update current model to be best model
func_rep.load_state_dict(trainer.best_model)

# Save full precision image reconstruction
with torch.no_grad():
    img_recon = func_rep(coordinates).reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1)
    save_image(torch.clamp(img_recon, 0, 1).to('cpu'), os.path.join(path, f'With_No_Grad_{args.image_id}_1.png'))

# Everything is the same so far as the train.py (with just one layer)
# Now it comes to the different part 
for i in range(args.num_layers-1):
    print(f'Now training model with {i+2} hidden layers...')
    func_rep = ExtendedSirenWithInsertion(func_rep, dim_hidden=args.layer_size).to(device)
    trainer = Trainer(func_rep, lr=2e-4)
    trainer.train(coordinates, features, num_iters=args.num_iters)
    print(f'Best training psnr: {trainer.best_vals["psnr"]:.2f}')
    torch.save(trainer.best_model, os.path.join(path, f'With_No_Grad_{args.image_id}_layer_{i+2}.pt'))
    with torch.no_grad():
        img_recon = func_rep(coordinates).reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1)
        save_image(torch.clamp(img_recon, 0, 1).to('cpu'), os.path.join(path, f'With_No_Grad_{args.image_id}_{i+2}.png'))



