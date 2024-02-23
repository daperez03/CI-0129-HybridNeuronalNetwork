from NeuralNetwork.Operations import vectors
import matplotlib.pyplot as plt
from PIL import Image
from tkinter import filedialog, Tk
import albumentations as A
from typing import Tuple

# Default image size
imageSize = (224, 224)

# Declare an augmentation pipeline
transform = A.Compose([
    A.RandomCrop(width=imageSize[0], height=imageSize[1]),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomRotate90(),
    A.Flip(),
    A.Transpose(),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    A.HueSaturationValue(p=0.3)
])


def readImage(path: str) -> Image:
    """
    Reads an image from the specified path.

    ### Parameters:
    1. path (str): The path to the image file.

    ### Returns:
    - Image: The image object.
    """
    image = Image.open(path)
    return image

def showImage(image):
    """
    Displays the given image using matplotlib.

    ### Parameters:
    1. image (vectors.ndarray): The image to display.
    """
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.show()


def isImage(name: str) -> bool:
    """
    Checks if a file name corresponds to an image file.

    ### Parameters:
    1. name (str): The file name to check.

    ### Returns:
    - bool: True if the file is an image, False otherwise.
    """
    return (
        name.find(".jpg") != -1
        or name.find(".png") != -1
        or name.find(".jpeg") != -1
    )


def isValid(image: Image) -> bool:
    """
    Checks if the given image is valid (has three dimensions).

    ### Parameters:
    1. image (Image): The image object to check.

    ### Returns:
    - bool: True if the image is valid, False otherwise.
    """
    return len(vectors.array(image).shape) == 3

def augmentateData(images, labels, augmentationFactor=1) -> Tuple[vectors.ndarray, vectors.ndarray]:
    """
    Augments the dataset by applying transformations to images.

    ### Parameters:
    1. images (vectors.ndarray): The array of images.
    2. labels (vectors.ndarray): The array of corresponding labels.
    3. augmentationFactor (int): The factor by which to augment data. Defaults to 1.

    ### Returns
    vectors.ndarray, vectors.ndarray: The augmented images and the corresponding labels
    """
    augmentateCount = 0
    augmentedImages = []
    augmentedLabels = []
    
    for image, label in zip(images, labels):
        augmented = [transform(image=image)['image'] for _ in range(augmentationFactor)]
        augmentedImages.extend(augmented)
        augmentedLabels.extend([label] * augmentationFactor)
        augmentateCount += 1

    print(f"Data augmentated! Augmentated items: {augmentateCount}")
    return vectors.array(augmentedImages), vectors.array(augmentedLabels)

def shuffleData(images, labels) -> Tuple[vectors.ndarray, vectors.ndarray]:
    """
    Shuffles images and labels arrays together.

    ### Parameters:
    1. images (vectors.ndarray): The array of images.
    2. labels (vectors.ndarray): The array of corresponding labels.

    ### Returns:
    - vectors.ndarray, vectors.ndarray: A tuple containing shuffled images and labels arrays.
    """
    images = vectors.array(images, dtype=vectors.float32)  # Convert images outside the function if possible
    labels = vectors.array(labels)

    # Generate random indices for shuffling
    indices = vectors.random.permutation(len(labels))

    # Shuffle both arrays using the same indices
    shuffled_images = images[indices]
    shuffled_labels = labels[indices]
    return vectors.array(shuffled_images), vectors.array(shuffled_labels)


def openDirectoryExplorer(initialDirectory) -> str:
    """
    Opens a directory explorer to select a file path.

    ### Returns:
    - str: The selected file path.
    """
    root = Tk()
    root.withdraw()  # Oculta la ventana principal

    file_path = filedialog.askopenfilename(initialdir=initialDirectory)

    return file_path
