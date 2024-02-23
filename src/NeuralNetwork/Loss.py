from scipy.stats import gmean
from typing import Tuple
from NeuralNetwork.Operations import vectors

# Base Loss
class Loss:
    """
    Base class for handling loss calculations.

    ### Methods
    1. `__init__()` : Initializes the Loss class.
    2. `clean()` : Resets the list of registered losses.
    3. `addRegisteredLoss(loss)` : Adds a loss value to the registered losses list.
    4. `calculateMean()` : Calculates the geometric mean of registered losses.
    5. `setLayers(layers)` : Sets the layers for loss calculation.
    6. `regularizationLoss()` : Calculates the regularization loss.
    7. `calculate(output, expectedResults)` : Calculates the loss given the output and expected results.
    """

    def __init__(self) -> None:
        """
        Initializes the Loss class.
        """
        self._registeredLosses = []

    def clean(self) -> None:
        """
        Resets the list of registered losses.
        """
        self._registeredLosses = []

    def addRegisteredLoss(self, loss) -> None:
        """
        Adds a loss value to the registered losses list.
        """
        self._registeredLosses.append(loss)

    def calculateMean(self) -> None:
        """
        Calculates the geometric mean of registered losses.
        """
        return gmean(self._registeredLosses)
    
    def setLayers(self, layers) -> "Loss":
        """
        Sets the layers for loss calculation.

        ### Parameters
        1. layers: Layers for loss calculation.
        """
        self._layers = layers
        return self
    
    def regularizationLoss(self) -> float:
        """
        Calculates the regularization loss.

        ### Returns
        - float: Regularization loss.
        """
        # 0 by default
        regularizationLoss = 0

        for layer in self._layers:
            if layer.isTrainable():
                # L1 regularization - weights
                if layer.weightRegularizerL1 > 0:
                    regularizationLoss += layer.weightRegularizerL1 * vectors.sum(vectors.abs(layer.getWeights()))

                # L2 regularization - weights
                if layer.weightRegularizerL2 > 0:
                    regularizationLoss += layer.weightRegularizerL2 * vectors.sum(layer.getWeights() * layer.getWeights())

                # L1 regularization - biases
                if layer.biasRegularizerL1 > 0:
                    regularizationLoss += layer.biasRegularizerL1 * vectors.sum(vectors.abs(layer.getBiases()))

                # L2 regularization - biases
                if layer.biasRegularizerL2 > 0:
                    regularizationLoss += layer.biasRegularizerL2 * vectors.sum(layer.getBiases() * layer.getBiases())

        return regularizationLoss

    def calculate(self, output, expectedResults) -> Tuple[float, float]:
        """
        Calculates the loss given the output and expected results.

        ### Parameters
        1. output (vectors.ndarray): The output values.
        2. expectedResults (vectors.ndarray or list): The expected result values.

        ### Returns
        - Tuple[float, float]: Data loss and regularization loss.
        """
        losses = self.forward(output, expectedResults)
        dataLoss = vectors.mean(losses)
        return dataLoss, self.regularizationLoss()

class Loss_CategoricalCrossentropy(Loss):
    """
    Calculates the categorical cross-entropy loss.

    Inherits from `Loss` class.

    ### Methods
    1. `__init__()`: Initializes the Loss_CategoricalCrossentropy class.
    2. `forward(resultsPredictions, expectedResults)`: Calculates the forward pass of categorical cross-entropy.
    3. `backward(dvalues, expectedResults)`: Calculates the backward pass of categorical cross-entropy.

    """

    def __init__(self) -> None:
        """
        Initializes the Loss_CategoricalCrossentropy class.
        """
        super().__init__()

    def forward(self, resultsPredictions, expectedResults) -> vectors.ndarray:
        """
        Calculates the forward pass of categorical cross-entropy.

        ### Parameters
        1. resultsPredictions (vectors.ndarray): Predicted results.
        2. expectedResults (vectors.ndarray): Expected results.

        ### Returns
        - vectors.ndarray: Calculated likelihoods.
        """
        # Number of samples in a batch
        samples = len(resultsPredictions)
        # Clip data to prevent division by 0
        resultsPredictionsClipped = vectors.clip(resultsPredictions, 1e-7, 1 - 1e-7)
        if len(expectedResults.shape) == 1:
            trueConfidences = resultsPredictionsClipped[range(samples), expectedResults]

        likelihoods = -vectors.log(trueConfidences)
        return likelihoods

    def backward(self, dvalues, expectedResults) -> None:
        """
        Calculates the backward pass of categorical cross-entropy.

        ### Parameters
        1. dvalues (vectors.ndarray): Gradient values.
        2. expectedResults (vectors.ndarray): Expected results.
        """
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(expectedResults.shape) == 1:
            expectedResults = vectors.eye(labels)[expectedResults]
        
        self.dinputs = -expectedResults / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples
