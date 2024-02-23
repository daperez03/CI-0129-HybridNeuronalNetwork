import enum
from NeuralNetwork.Operations import vectors

from scipy.signal import convolve2d

# Various predefined filters
point = [
    [-0.627, 0.352, -0.627],
    [0.352, 2.923, 0.352],
    [-0.627, 0.352, -0.627]]

bw = [
    [0.299, 0.587, 0.114],
    [0.299, 0.587, 0.114],
    [0.299, 0.587, 0.114]
]

# Default filter
default_filter = bw

class FilterType(enum.Enum):
    """
    Enumerates different types of filters.
    """
    Point = 1
    BW = 2
  
def applyConvolution(image, filter) -> list:
    """
    Applies convolution operation on the input image with the specified filter.

    ### Parameters
    1. image (vectors.ndarray): The input image as a NumPy array.
    2. filter (list): The convolution filter.

    ### Returns
    - list: The convoluted image as a list of lists.
    """
    newImage = []
    iter1 = 0
    size = len(image) * len(image[0])
    for iter in range(size):
        iter1 = iter // len(image[0])
        iter2 = iter % len(image[0])
        if iter2 == 0:
            newImage.append([])
        if iter1 == 0 or iter2 == 0 or iter1 == len(image) - 1 or iter2 == len(image[iter1]) - 1:
            newImage[iter1].append(image[iter1][iter2])
        else:
            # Apply convolution with the filter to each color channel
            result = [
                vectors.multiply(image[iter1 - 1 : iter1 + 2, iter2 - 1 : iter2 + 2, 0], filter),
                vectors.multiply(image[iter1 - 1 : iter1 + 2, iter2 - 1 : iter2 + 2, 1], filter),
                vectors.multiply(image[iter1 - 1 : iter1 + 2, iter2 - 1 : iter2 + 2, 2], filter)
            ]
            # Sum the results across color channels and convert to integer
            result = vectors.array(result).sum(axis=(1, 2)).astype(int)

            newImage[iter1].append(vectors.array(result) % 255)
        
    return newImage

def applyConvolutionEfficient(images, filter) -> list:
    """
    Applies convolution operation on multiple images with the specified filter in an optimized way.

    ### Parameters
    1. images (list): List of input images as NumPy arrays.
    2. filter (list): The convolution filter.

    ### Returns
    - list: List of convoluted images as NumPy arrays.
    """
    results = []
    for image in images:
        # Apply convolution on each color channel separately
        result_channel_0 = convolve2d(image[:, :, 0], filter, mode='same', boundary='fill', fillvalue=0.0)
        result_channel_1 = convolve2d(image[:, :, 1], filter, mode='same', boundary='fill', fillvalue=0.0)
        result_channel_2 = convolve2d(image[:, :, 2], filter, mode='same', boundary='fill', fillvalue=0.0)

        # Stack the results along the last axis to get the desired shape (224, 224, 3)
        result = vectors.stack([result_channel_0, result_channel_1, result_channel_2], axis=-1)
        results.append(result)

    return results

# Mapping of filter types to respective filter arrays
filterTypeMapping = {
    FilterType.Point: point,
    FilterType.BW: bw
}
