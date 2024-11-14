from scipy.stats import norm, laplace
import random 
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
import seaborn as sns 

def plot_weight_dist(all_weights, save_path):
    # Fit a Gaussian distribution to the data
    mu_gauss, std_gauss = norm.fit(all_weights)

    # Fit a Laplacian distribution to the data
    mu_laplace, b_laplace = laplace.fit(all_weights)

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_weights, bins=2000, density=True, color='blue', alpha=0.6, label='Weights Histogram')

    # Create an array of values for the x-axis (for plotting the PDFs)
    x = np.linspace(-0.05, 0.05, 1000)

    # Plot the Gaussian fit
    pdf_gauss = norm.pdf(x, mu_gauss, std_gauss)
    plt.plot(x, pdf_gauss, 'r-', linewidth=2, label='Gaussian fit')

    # Plot the Laplacian fit
    pdf_laplace = laplace.pdf(x, mu_laplace, b_laplace)
    plt.plot(x, pdf_laplace, 'g-', linewidth=2, label='Laplacian fit')

    # Add labels and title
    plt.title("Histogram of Neural Network Weights with Gaussian and Laplacian Fits")
    plt.xlabel("Weight Value")
    plt.ylabel("Density")
    plt.xlim(-0.05, 0.05)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path+'/weight_plot_histogram.png')

def create_gif_from_images(image_id, image_folder, output_gif, fps=10):
    """
    Create a GIF from a series of images in a folder.

    Args:
        image_folder (str): The path to the folder containing images.
        output_gif (str): The output path and filename for the GIF.
        fps (int): Frames per second (speed of the GIF).
    """
    # Get a sorted list of all images in the directory
    image_files = sorted([f for f in os.listdir(image_folder) if f.startswith('SuRP') and f.endswith('.png')],
                         key=lambda x: int(x.replace(f'SuRP_reconstruction_{image_id}_', '').replace('.png', '')))
    
    # Check if there are images to process
    if not image_files:
        print("No images found in the directory!")
        return
    
    # Read and store images
    images = []
    residues = [] 
    output_residues_gif = image_folder + '/residues'
    if not os.path.exists(output_residues_gif):
        os.makedirs(output_residues_gif)
    for i, filename in enumerate(image_files):
        filepath = os.path.join(image_folder, filename)
        images.append(imageio.imread(filepath))

        if i>0:
          residue = np.abs(images[i].astype(np.int16) - images[i-1].astype(np.int16))
          residue_filename = os.path.join(output_residues_gif, f"residue_{i}.png")
          imageio.imwrite(residue_filename, residue.astype(np.uint8))
          residues.append(residue.astype(np.uint8))  # Store for creating the GIF
  
    # Save images as a GIF
    imageio.mimsave(output_gif, images, fps=fps)
    print(f"GIF saved at {output_gif}")

    # Save residue images as a GIF
    imageio.mimsave(os.path.join(output_residues_gif, "result_animation.gif"), residues, fps=fps)
    print(f"Residue GIF saved at {output_residues_gif}")

def plot_psnr_sparsity(iters, spars, psnrs, save_path):
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
        plt.savefig(os.path.join(save_path, "sparsity_psnr_plot.png"))
        print(f"Plot saved at {os.path.join(save_path, 'sparsity_psnr_plot.png')}")
        # plt.show()

def plot_ssim_sparsity(iters, spars, ssims, save_path):

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
    ax2.plot(iters, ssims, marker='o', linestyle='-', color='r', label='MS-SSIMS')
    ax2.set_ylabel('MS-SSIMS', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Set title for the plot
    plt.title('Iterations vs. Sparsity and MS-SSIMS')

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # Adjust layout for better spacing
    fig.tight_layout()

    # Save the plot to the specified path
    plt.savefig(os.path.join(save_path, "sparsity_ssims_plot.png"))
    print(f"Plot saved at {os.path.join(save_path, 'sparsity_psnr_plot.png')}")
    # plt.show()

def plot_recon_loc(layer_nums, num_layers, save_path):
  # Create an empty 2D array to count reconstructions per layer per iteration
    num_iterations = len(layer_nums)
    reconstruction_counts = np.zeros((layer_nums, num_iterations))

    # Fill in the counts based on the layer data
    for i, layers in enumerate(layer_nums):
        for layer in layers:
            reconstruction_counts[layer - 1, i] += 1  # Assuming layer indices are 1-based

    # Plotting the heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(reconstruction_counts, cmap="YlGnBu", annot=False, cbar=True, 
                xticklabels=100 if num_iterations > 100 else 10, yticklabels=1)
    plt.title("Reconstruction Heatmap: Layer vs Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Layer")
    # plt.show()
    print(f"Plot saved at {os.path.join(save_path, 'weights_loc_per_iter.png')}")




