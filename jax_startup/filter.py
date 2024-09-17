import psutil
# Get the number of physical CPUs
num_physical_cpus = psutil.cpu_count(logical=False)

import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=%d'%num_physical_cpus

import numpy as np
import jax
from jax import lax
from jax import numpy as jnp

def gaussian_kernal(P, sigma=1.0):
    """
    Generates an (2P+1) x (2P+1) Gaussian kernel.

    This function creates a Gaussian kernel, which is a square matrix of size N x N.
    Each element of the matrix is computed using the Gaussian function, based on its distance from the center of the matrix.

    Parameters
    ----------
    N : int
        The size (width and height) of the output Gaussian kernel. It determines the diameter of the filter.
    sigma : float, optional
        The standard deviation of the Gaussian distribution. It controls the spread or width of the Gaussian function. Defaults to 1.0.

    Returns
    -------
    numpy.ndarray
        An N x N matrix representing the Gaussian kernel.
    """

    N = 2*P +1
    coords = np.arange(N) - N // 2
    X, Y = np.meshgrid(coords, coords)
    kernel = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()

def dist_kernel(P):
    """
    Generates an (2P+1) x (2P+1) distance kernel.

    This function creates a Gaussian kernel, which is a square matrix of size N x N.
    Each element of the matrix is computed using the Gaussian function, based on its distance from the center of the matrix.

    Parameters
    ----------
    N : int
        The size (width and height) of the output Gaussian kernel. It determines the diameter of the filter.
    sigma : float, optional
        The standard deviation of the Gaussian distribution. It controls the spread or width of the Gaussian function. Defaults to 1.0.

    Returns
    -------
    numpy.ndarray
        An N x N matrix representing the Gaussian kernel.
    """

    N = 2*P +1
    coords = np.arange(N) - N // 2
    X, Y = np.meshgrid(coords, coords)
    kernel = (X ** 2 + Y ** 2)
    kernel = np.where(kernel != 0, 1.0/kernel, kernel)
    return kernel / kernel.sum()

# Numpy filtering
def FIR_filter_np(image, kernel):
    """
    NumPy implementatin of an FIR filter to an image using zero-padding with "same" boundary condition.

    This function applies an FIR filter(implemented by NumPy) with given kernel to an input grayscale image using convolution.
    The kernel is slid over the image to calculate the sum of the element-wise multiplication
    between the kernel and the portion of the image it currently covers.

    Parameters
    ----------
    image : numpy.ndarray
        The input grayscale image to which the filter is to be applied.
    kernel : numpy.ndarray
        The kernel to be applied on the image.

    Returns
    -------
    numpy.ndarray
        The filtered image after applying the kernel.
    """

    # Determine value of P so that kernel is (2P+1)x(2P+1)
    P = kernel.shape[0] // 2

    # Initialize output array with zeros, same shape as input image
    output = np.zeros_like(image)

    # Zero pad the input image using a boundary of width P
    pad_img = np.pad(image, [(P, P), (P, P)])

    # Apply filtering
    for i in range(output.shape[0]):
        # Print progress for every 100 rows processed
        if i % 100 == 0:
            percentage = (i / output.shape[0]) * 100
            print(f"\rProgress: {percentage:.2f}%", end='')

        for j in range(output.shape[1]):
            # Convolve kernel with image
            output[i, j] = np.sum(pad_img[i:i + kernel.shape[0], j:j + kernel.shape[1]] * kernel)
    print()
    return output


# Jax vmap filtering
def FIR_filter_jax(image, kernal, map_method='vmap'):
    """
    Applies a filter to an image using JAX.

    This function applies an FIR filter(implemented by JAX) with given kernel to an input grayscale image using convolution.
    The kernel is slid over the image to calculate the sum of the element-wise multiplication
    between the kernel and the portion of the image it currently covers.

    Parameters
    ----------
    image : numpy.ndarray
        The input grayscale image to which the filter is to be applied.
    kernal : numpy.ndarray
        The kernal to be applied on the image.

    Returns
    -------
    numpy.ndarray
        The filtered image after applying the filter.
    """

    # Convert input image and kernal to JAX arrays
    image = jnp.array(image)
    kernal = jnp.array(kernal)

    # Store shape of kernal for later use in convolution
    kernal_shape = kernal.shape

    # Pad the input image to handle edges during convolution
    pad_image = jnp.pad(image, [(kernal_shape[0] // 2, kernal_shape[0] // 2), (kernal_shape[1] // 2, kernal_shape[1] // 2)],
                        mode='constant')

    def conv_at_ij(i, j):
        # Extract patch from padded image at position (i, j)
        patch = lax.dynamic_slice(pad_image, (i, j), kernal_shape)

        # Return sum of element-wise multiplication of patch and filter
        return jnp.sum(patch * kernal)

    # Create meshgrid of indices for input image
    I, J = np.meshgrid(np.arange(image.shape[0]),
                       np.arange(image.shape[1]), indexing="ij")

    if map_method == 'vmap':
        # Use vmap to compute the output of the convolution at each point in (I,J),
        # then reshape into the original image size
        result = jax.vmap(conv_at_ij)(I.flatten(), J.flatten()).reshape(image.shape[0], image.shape[1])

    elif map_method == 'pmap':
        length = len(I.flatten())
        sqrt_npc = int(np.sqrt(num_physical_cpus))
        k = int(np.ceil(length / sqrt_npc))
        I = np.resize(I, (sqrt_npc, k))
        J = np.resize(J, (sqrt_npc, k))

        # Apply convolution function to each position in the image and reshape output to original image shape
        result = jax.pmap(jax.vmap(conv_at_ij))(I, J)
        result = result.flatten()[:length].reshape(image.shape[0], image.shape[1])
    else:
        raise ValueError('map_method should be \'vmap\' or \'pmap\'')
    # Apply convolution function to each position in the image and reshape output to original image shape
    return result
