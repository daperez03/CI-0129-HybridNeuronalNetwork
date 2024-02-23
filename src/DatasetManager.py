from NeuralNetwork.Operations import vectors
import os
import NeuralNetwork.Utils as Utils
from typing import Tuple

labelsMapping = {
    "apple": 0,
    "avocado": 1,
    "banana": 2,
    "cherry": 3,
    "kiwi": 4,
    "mango": 5,
    "orange": 6,
    "pineapple": 7,
    "strawberries": 8,
    "watermelon": 9
}

# Invert the dictionary
labelsInvertedMapping = {v: k for k, v in labelsMapping.items()}

def directoryIterator(rootPath=".") -> str:
    """
    Iterates through files and directories in a given root path.

    ### Parameters:
    1. rootPath (str): The root path to iterate through. Defaults to the current directory.

    ### Yields:
    - str: Paths to files or directories.
    """
    for root, dirs, files in os.walk(rootPath):
        for file in files:
            yield os.path.join(root, file)
        for dir in dirs:
            yield os.path.join(root, dir)


def mapFruit(name: str) -> int:
    """
    Maps a fruit name to its corresponding label index.

    ### Parameters:
    1. name (str): The name of the fruit.

    ### Returns:
    - int: The label index of the fruit.
    """
    for label in labelsMapping:
        if name.find(label) != -1:
            return labelsMapping[label]
    return -1

def fruitsData(typeOfData: str, augmentateFactor = 0, shuffle=False) -> Tuple[vectors.ndarray, vectors.ndarray]:
    """
    Loads fruit dataset images and labels.

    ### Parameters:
    1. typeOfData (str): The type of data to load.
    2. augmentate (bool): Indicate if augmentation should be applied

    ### Returns:
    - Tuple[vectors.ndarray, vectors.ndarray]: A tuple containing shuffled images and labels arrays.
    """
    rootDirectory = os.path.join(os.path.dirname(__file__), "../Datasets/" + typeOfData)

    images: list = []
    labels: list = []

    imageCounter = 0
    for item in directoryIterator(rootDirectory):
        fruit = mapFruit(item)
        if fruit != -1 and Utils.isImage(item):
            image = Utils.readImage(item)
            if Utils.isValid(image):
                image = image.resize(Utils.imageSize)
                images.append(vectors.array(image))
                labels.append(fruit)
                imageCounter += 1

    images = vectors.array(images)
    labels = vectors.array(labels)
  
    if augmentateFactor:
        augmentedImages, augmentedLabels = Utils.augmentateData(images, labels, augmentateFactor)
        images, labels =  vectors.concatenate((images, augmentedImages), axis=0), vectors.concatenate((labels, augmentedLabels))
   
    if shuffle:
        images, labels = Utils.shuffleData(images, labels)
    
    print(f"Dataset read! Items read: {imageCounter}")
    return images, labels
