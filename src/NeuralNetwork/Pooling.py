from NeuralNetwork.Operations import vectors
from skimage.measure import block_reduce
from multiprocessing import Pool
import enum

class PoolType(enum.Enum):
    """
    Enumerates various pooling types.
    """
    MAX = 1
    AVG = 2
    MIN = 3

def avg_pool(block, axis):
    return vectors.round(vectors.mean(block, axis=axis))

poolingTypeMapping = {
    PoolType.MAX: vectors.max,
    PoolType.AVG: avg_pool,
    PoolType.MIN: vectors.min
}

workImages = []

# Default pool type
poolType = PoolType.MAX
poolDimensions = (2, 2)

def applyPooling(image, type: PoolType, poolDimension=(2, 2)) -> list:
    """
    Applies pooling to the input image based on the specified pooling type and dimensions.

    ### Parameters
    - image (vectors.ndarray): The input image.
    - type (PoolType): The pooling type (MAX, AVG, or MIN).
    - poolDimension (tuple): The pooling window dimensions.

    ### Returns
    - list: The pooled image.
    """
    newImage = []
    if len(image) % poolDimension[0] == 0 and len(image[0]) % poolDimension[1] == 0:
        iter1 = 0
        while iter1 < len(image):
            newImage.append([])
            iter2 = 0
            while iter2 < len(image[iter1]):
                poolMatrix = []
                for iter3 in range(poolDimension[0]):
                    poolMatrix.append([])
                    for iter4 in range(poolDimension[1]):
                        poolMatrix[iter3].append(image[iter1 + iter3][iter2 + iter4])
                result = []
                if type == PoolType.AVG:
                    result = vectors.mean(poolMatrix)
                elif type == PoolType.MAX:
                    result = vectors.max(poolMatrix)
                newImage[len(newImage)-1].append(result)
                iter2 += poolDimension[1]
            iter1 += poolDimension[0]
    else:
        raise Exception("Image cannot be transformed with this pool dimensions")
    return newImage

def poolImage(image, function, dimensions):
    return block_reduce(image, dimensions, func=function)

def applyPoolingEfficient(images, poolType: PoolType, dimensions) -> list:
    """
    Applies efficient pooling to a list of images based on the specified pooling type and dimensions.

    ### Parameters
    - images (list): List of input images.
    - poolType (PoolType): The pooling type (MAX, AVG, or MIN).
    - dimensions (tuple): The pooling window dimensions.

    ### Returns
    - list: The list of pooled images.
    """
    function = poolingTypeMapping[poolType]
    
    with Pool() as pool:
        results = pool.starmap(poolImage, [(image, function, dimensions) for image in images])
    
    # Explicitly close the pool to stop accepting new tasks.
    pool.close()

    # Wait for all tasks to complete
    pool.join()

    return results
