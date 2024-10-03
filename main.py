import argparse
import getpass
import imageio
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
parser.add_argument("-ld", "--logdir", help="Path to save logs", default="results")
parser.add_argument("-ni", "--num_iters", help="Number of iterations to train for", type=int, default=50000)
parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=2e-4)
parser.add_argument("-iid", "--image_id", help="Image ID to train on, if not the full dataset", type=int, default=15)
parser.add_argument("-lss", "--layer_size", help="Layer sizes as list of ints", type=int, default=28)
parser.add_argument("-nl", "--num_layers", help="Number of layers", type=int, default=10)
parser.add_argument("-w0", "--w0", help="w0 parameter for SIREN model.", type=float, default=30.0)
parser.add_argument("-w0i", "--w0_initial", help="w0 parameter for first layer of SIREN model.", type=float, default=30.0)
## For magnitude pruning
parser.add_argument("-pr", "--prune_ratio", help ="pruning ratio", type = float, default = 0.4)
parser.add_argument("-ri","--refine_iter", help = "number of refine iterations", type = int, default = 1000)
## For loading trained model? 


args = parser.parse_args()

# Dictionary to register mean values (both full precision and half precision)
# results = {'fp_bpp': [], 'hp_bpp': [], 'fp_psnr': [], 'hp_psnr': []}
results = {'fp_bpp': [], 'fpp_bpp':[], 'hp_bpp': [], 
           'fp_psnr': [], 'fpp_psnr': [], 'hp_psnr': []}

# Create directory to store experiments
if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)

# Fit images
for i in range(min_id, max_id + 1):
    print(f'Image {i}')

    # Load image
    img = imageio.imread(f"kodak-dataset/kodim{str(i).zfill(2)}.png")
    img = transforms.ToTensor()(img).float().to(device, dtype)

    # Setup model
    func_rep = Siren(
        dim_in=2,
        dim_hidden=args.layer_size,
        dim_out=3,
        num_layers=args.num_layers,
        final_activation=torch.nn.Identity(),
        w0_initial=args.w0_initial,
        w0=args.w0
    ).to(device)

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

    # Log full precision results
    results['fp_bpp'].append(fp_bpp)
    results['fp_psnr'].append(trainer.best_vals['psnr'])

    # Save best model
    torch.save(trainer.best_model, args.logdir + f'/best_model_{i}.pt')

    # Update current model to be best model
    func_rep.load_state_dict(trainer.best_model)

    # Save full precision image reconstruction
    with torch.no_grad():
        img_recon = func_rep(coordinates).reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1)
        save_image(torch.clamp(img_recon, 0, 1).to('cpu'), args.logdir + f'/fp_reconstruction_{i}.png')
    
    # Extract weights for plotting 
    # all_weights = util.extract_weights(func_rep)

    # Prune Model and refine 
    func_rep_pruned, masks = util.apply_magnitude_pruning(func_rep, pruning_percent=args.prune_ratio)
    trainer = Trainer(func_rep_pruned, lr=1e-3, sparse_training=True, masks=masks)
    trainer.train(coordinates, features, num_iters=args.refine_iter)
    print(f'Best training psnr: {trainer.best_vals["psnr"]:.2f}')

    # Calculate model size. Divide by 8000 to go from bits to kB
    model_size = util.model_size_in_bits(func_rep) / 8000.
    print(f'Model size: {model_size:.1f}kB')
    fpp_bpp = util.bpp(model=func_rep, image=img)
    print(f'Full precision pruned bpp: {fpp_bpp:.2f}')

    # Log full precision pruend results
    results['fpp_bpp'].append(fpp_bpp)
    results['fpp_psnr'].append(trainer.best_vals['psnr'])

    # Save best model
    torch.save(trainer.best_model, args.logdir + f'/best_pruned_model_{i}.pt')

    # Update current model to be best model
    func_rep_pruned.load_state_dict(trainer.best_model)

    # Save full precision image reconstruction
    with torch.no_grad():
        img_recon = func_rep_pruned(coordinates).reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1)
        save_image(torch.clamp(img_recon, 0, 1).to('cpu'), args.logdir + f'/fpp_reconstruction_{i}.png')

    # Convert model and coordinates to half precision. Note that half precision
    # torch.sin is only implemented on GPU, so must use cuda
    if torch.cuda.is_available():
        func_rep = func_rep.half().to('cuda')
        coordinates = coordinates.half().to('cuda')
        torch.save()

        # Calculate model size in half precision
        hp_bpp = util.bpp(model=func_rep, image=img)
        results['hp_bpp'].append(hp_bpp)
        print(f'Half precision bpp: {hp_bpp:.2f}')

        # Compute image reconstruction and PSNR
        with torch.no_grad():
            img_recon = func_rep(coordinates).reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1).float()
            hp_psnr = util.get_clamped_psnr(img_recon, img)
            save_image(torch.clamp(img_recon, 0, 1).to('cpu'), args.logdir + f'/hp_reconstruction_{i}.png')
            print(f'Half precision psnr: {hp_psnr:.2f}')
            results['hp_psnr'].append(hp_psnr)
    else:
        results['hp_bpp'].append(fp_bpp)
        results['hp_psnr'].append(0.0)

    # Save logs for individual image
    with open(args.logdir + f'/logs{i}.json', 'w') as f:
        json.dump(trainer.logs, f)

    print('\n')

print('Full results:')
print(results)
with open(args.logdir + f'/results.json', 'w') as f:
    json.dump(results, f)

# Compute and save aggregated results
results_mean = {key: util.mean(results[key]) for key in results}
with open(args.logdir + f'/results_mean.json', 'w') as f:
    json.dump(results_mean, f)

print('Aggregate results:')
print(f'Full precision, bpp: {results_mean["fp_bpp"]:.2f}, psnr: {results_mean["fp_psnr"]:.2f}')
print(f'Full precision, bpp: {results_mean["fpp_bpp"]:.2f}, psnr: {results_mean["fpp_psnr"]:.2f}')
print(f'Half precision, bpp: {results_mean["hp_bpp"]:.2f}, psnr: {results_mean["hp_psnr"]:.2f}')


# Plot Weight Distribution 
# util.plot_weight_dist(all_weights)

# TODO: Implement SuRP (Changed: will implement in sr.py)
# SR_INR = surp(func_rep, beta = 1, total_iter=1000, width = args.layer_size, depth = args.layer_depth)
# SR_INR.successive_refine()


