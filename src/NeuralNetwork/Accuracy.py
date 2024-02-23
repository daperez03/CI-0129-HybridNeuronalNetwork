from NeuralNetwork.Operations import vectors

from scipy.stats import gmean
from abc import ABC, abstractclassmethod

class Accuracy(ABC):
    """
    Handles accuracy calculations for predictions.

    ### Methods
    1. `__init__()`: Initializes the Accuracy class.
    2. `clean()`: Resets the list of registered accuracies.
    3. `addRegisteredAccuracy(accuracy)`: Adds an accuracy value to the registered accuracies list.
    4. `calculateMean()`: Calculates the geometric mean of registered accuracies.
    5. `calculate(predictions, expectedOutput)`: Calculates the accuracy between predictions and expected output.
    """

    def __init__(self) -> None:
        """
        Initializes the Accuracy class.
        """
        self._registeredAccuracies = []

    def clean(self) -> None:
        """
        Resets the list of registered accuracies.
        """
        self._registeredAccuracies = []

    def addRegisteredAccuracy(self, accuracy) -> None:
        """
        Adds an accuracy value to the registered accuracies list.
        """
        self._registeredAccuracies.append(accuracy)

    def calculateMean(self) -> float:
        """
        Calculates the geometric mean of registered accuracies.

        ### Returns
        - float: Geometric mean of registered accuracies
        """
        return gmean(self._registeredAccuracies)

    def calculate(self, predictions, expectedOutput):
        """
        Calculates the accuracy between predictions and expected output.

        ### Parameters
        1. predictions (vectors.ndarray): The predicted values.
        2. expectedOutput (vectors.ndarray): The expected output values.

        ### Returns
        - float: Calculated accuracy.
        """
        comparisons = self.compare(predictions, expectedOutput)
        accuracy = vectors.mean(comparisons)
        return accuracy

    @abstractclassmethod
    def compare(self, predictions, expectedOutput):
        """
        Compares the predictions to the exceptedOutput

        ### Parameters
        1. predictions (vectors.ndarray): The predicted values.
        2. expectedOutput (vectors.ndarray): The expected output values.

        ### Raises
        - NotImplementedError: Subclasses must implement the `compare` method.
        """
        raise NotImplementedError("Subclasses must implement compare method")

class Accuracy_Categorical(Accuracy):
    """
    A subclass of `Accuracy` that deals specifically with categorical accuracy calculations.

    ### Methods
    1. `__init__()`: Initializes the Accuracy_Categorical class.
    2. `compare(predictions, expectedOutput)`: Compares categorical predictions with expected output.
    """

    def __init__(self) -> None:
        super().__init__()

    def compare(self, predictions, expectedOutput) -> vectors.ndarray:
        """
        Compares categorical predictions with expected output.

        ### Parameters
        1. predictions (vectors.ndarray): The predicted values.
        2. expectedOutput (vectors.ndarray): The expected output values.

        ### Returns
        - vect.ndarray: Boolean array indicating the correctness of predictions.
        """
        if len(predictions.shape) == 2:
            predictions = vectors.argmax(predictions, axis=1)
        return predictions == expectedOutput
