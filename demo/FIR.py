import numpy as np
from PIL import Image
import jax
import time
import argparse
import os
import matplotlib.pyplot as plt

from jax_startup.filter import gaussian_kernal, FIR_filter_np, FIR_filter_jax


# Define Gaussian filter



if __name__ == '__main__':
    # Set up argument parser for image, which needs to be filtered.
    parser = argparse.ArgumentParser(description="Apply Gaussian filter to image using numpy and jax.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    args = parser.parse_args()

    # Extract the filename from provided path
    # Splits the filename into the base filename and its extension.
    filename, _ = os.path.splitext(os.path.basename(args.image_path))

    # Load the image and convert to grayscale
    image = np.array(Image.open(args.image_path).convert('L')).astype('float')

    # Set filter order
    P = 5
    N = 2*P +1

    # Generate Gaussian filter
    kernel = gaussian_kernal(P, sigma=5)

    # Run FIR filtering.
    # Time and apply filtering using NumPy
    start_time = time.time()
    numpy_output = FIR_filter_np(image, kernel)
    numpy_duration = time.time() - start_time
    print(f"Numpy filter time: {numpy_duration:.4f} seconds")

    # Time and apply filtering using JAX
    start_time = time.time()
    jax_output_vmap = jax.jit(FIR_filter_jax, device=jax.devices('cpu')[0])(image, kernel)
    jax_duration = time.time() - start_time
    print(f"Jax filter with vmap time: {jax_duration:.4f} seconds")

    # Time and apply filtering using JAX
    start_time = time.time()
    jax_output_pmap = FIR_filter_jax(image, kernel, map_method='pmap')
    jax_duration = time.time() - start_time
    print(f"Jax filter with pmap time: {jax_duration:.4f} seconds")

    # Save result figures.
    # Make an output folder and save both ground truth and filtered figures.
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    print("Save to %s" % output_dir)
    Image.fromarray(image.astype(np.uint8)).save(os.path.join(output_dir, filename + "_gt.png"))
    Image.fromarray(numpy_output.astype(np.uint8)).save(os.path.join(output_dir, filename + "_numpy_output.png"))
    Image.fromarray(np.array(jax_output_vmap).astype(np.uint8)).save(os.path.join(output_dir, filename + "_jaxvmap_output.png"))
    Image.fromarray(np.array(jax_output_pmap).astype(np.uint8)).save(os.path.join(output_dir, filename + "_jaxpmap_output.png"))

    # Display
    # Determine the size of the patch you want to display
    patch_size_y = image.shape[0] // 8
    patch_size_x = image.shape[1] // 8

    # Compute starting and ending indices for rows (i.e., y-coordinates)
    y_start = image.shape[0] // 2 - patch_size_y // 2
    y_end = y_start + patch_size_y

    # Compute starting and ending indices for columns (i.e., x-coordinates)
    x_start = image.shape[1] // 2 - patch_size_x // 2
    x_end = x_start + patch_size_x

    # Display results using Matplotlib
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(image.astype(np.uint8)[y_start:y_end, x_start:x_end], cmap='gray')
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    axes[0, 1].imshow(numpy_output.astype(np.uint8)[y_start:y_end, x_start:x_end], cmap='gray')
    axes[0, 1].set_title("Filtered (NumPy)")
    axes[0, 1].axis('off')
    axes[1, 0].imshow(np.array(jax_output_vmap).astype(np.uint8)[y_start:y_end, x_start:x_end], cmap='gray')
    axes[1, 0].set_title("Filtered (JAX-Vmap)")
    axes[1, 0].axis('off')
    axes[1, 1].imshow(np.array(jax_output_pmap).astype(np.uint8)[y_start:y_end, x_start:x_end], cmap='gray')
    axes[1, 1].set_title("Filtered (JAX-Pmap)")
    axes[1, 1].axis('off')
    plt.tight_layout()
    plt.show()

    print(f"Outputs saved in {output_dir}")
