from NeuralNetwork.Operations import vectors
import enum
from abc import ABC, abstractmethod
import NeuralNetwork.Loss as Loss

class ActivationType(enum.Enum):
    """
    Enumerates various activation types.
    """
    Activation_ReLU = 1
    Activation_Softmax = 2
    Activation_Sigmoid = 3
    Activation_Tanh = 4
    Activation_Softmax_Loss_CategoricalCrossentropy = 5

class Activation(ABC):
    """
    Base class for different activation functions.

    ### Methods
    1. `__init__(inputs)`: Initializes the class.
    2. `forward(inputs)`: Performs forward pass.
    3. `backward(dvalues)`: Performs backward pass.
    4. `predictions(outputs)`: Calculates predictions.
    """
    def __init__(self) -> None:
        self.inputs = None
        self.output = None

    @abstractmethod
    def getEnum(self) -> ActivationType:
        """
        Gets the enum type

        ### Returns:
        - ActivationType: Activation type enum
        """
        raise NotImplementedError("Subclasses must implement the getEnum method")

    def forward(self, inputs) -> vectors.ndarray:
        """
        Performs the forward pass.

        ### Parameters
        1. inputs (vectors.ndarray): The input values.

        ### Returns
        - vectors.ndarray: Output after applying the activation function.
        """
        self.inputs = inputs
        return self.output

    @abstractmethod
    def backward(self, dvalues) -> None:
        """
        Performs the backward pass.

        ### Parameters
        1. dvalues (vectors.ndarray): The gradient of the loss with respect to the inputs.
        
        ### Raises
        - NotImplementedError: Method must be implemented in the subclass.
        """
        raise NotImplementedError("Method must be implemented in the subclass")

    @abstractmethod
    def predictions(self, outputs):
        """
        Calculates predictions.

        ### Parameters
        1. outputs (vectors.ndarray): The output values.

        ### Raises
        - NotImplementedError: Method must be implemented in the subclass.
        """
        raise NotImplementedError("Method must be implemented in the subclass")

# Actiation ReLU
class Activation_ReLU(Activation):
    """
    Activation function ReLU.

    Inherits from Activation class.

    ### Methods
    1. `__init__(inputs)`: Initializes the class.
    2. `forward(inputs)`: Performs forward pass.
    3. `backward(dvalues)`: Performs backward pass.
    4. `predictions(outputs)`: Calculates predictions.
    """

    def __init__(self) -> None:
        """
        Initializes the Activation ReLU class.
        """
        super().__init__()

    def getEnum(self) -> ActivationType:
        """
        Gets the enum type

        ### Returns:
        - ActivationType: Activation type enum
        """
        return ActivationType.Activation_ReLU
    
    def forward(self, inputs) -> vectors.ndarray:
        """
        Performs the forward pass for ReLU.

        ### Parameters
        1. inputs (vectors.ndarray): The input values.

        ### Returns
        - vectors.ndarray: Output after applying ReLU.
        """
        super().forward(inputs)
        self.output = vectors.maximum(0, inputs)
        return self.output

    def backward(self, dvalues) -> None:
        """
        Performs the backward pass for ReLU.

        Inherits from Activation class.

        ### Parameters
        1. dvalues (vectors.ndarray): The gradient of the loss with respect to the inputs.
        """
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs) -> vectors.ndarray:
        """
        Calculates predictions for ReLU.

        ### Parameters
        1. outputs (vectors.ndarray): The output values.

        ### Returns
        - vectors.ndarray: Predicted values based on the ReLU activation.
        """
        return outputs

# Activation Segmoid
class Activation_Sigmoid(Activation):
    """
    Activation function Sigmoid.

    Inherits from Activation class.

    ### Methods
    1. `__init__(inputs)`: Initializes the class.
    2. `forward(inputs)`: Performs forward pass.
    3. `backward(dvalues)`: Performs backward pass.
    4. `predictions(outputs)`: Calculates predictions.
    """

    def __init__(self) -> None:
        """
        Initializes the Activation Sigmoid class.
        """
        super().__init__()
    
    def getEnum(self) -> ActivationType:
        """
        Gets the enum type

        ### Returns:
        - ActivationType: Activation type enum
        """
        return ActivationType.Activation_Sigmoid
    
    def forward(self, inputs) -> vectors.ndarray:
        """
        Performs the forward pass for Sigmoid.

        ### Parameters
        1. inputs (vectors.ndarray): The input values.

        ### Returns
        - vectors.ndarray: Output after applying Sigmoid.
        """
        super().forward(inputs)
        self.output = 1 / (1 + vectors.exp(-inputs))
        return self.output

    def backward(self, dvalues) -> None:
        """
        Performs the backward pass for Sigmoid.

        ### Parameters
        1. dvalues (vectors.ndarray): The gradient of the loss with respect to the inputs.
        """
        self.dinputs = dvalues * (self.output * (1 - self.output))

    def predictions(self, outputs) -> vectors.ndarray:
        """
        Calculates predictions for Sigmoid.

        ### Parameters
        1. outputs (vectors.ndarray): The output values.

        ### Returns
        - vectors.ndarray: Predicted values based on the Sigmoid activation.
        """
        return (outputs > 0.5) * 1

# Activation Tanh
class Activation_Tanh(Activation):
    """
    Activation function Tanh.

    Inherits from Activation class.

    ### Methods
    1. `__init__(inputs)`: Initializes the class.
    2. `forward(inputs)`: Performs forward pass.
    3. `backward(dvalues)`: Performs backward pass.
    4. `predictions(outputs)`: Calculates predictions.
    """

    def __init__(self) -> None:
        """
        Initializes the Activation Tanh class.
        """
        super().__init__()

    def getEnum(self) -> ActivationType:
        """
        Gets the enum type

        ### Returns:
        - ActivationType: Activation type enum
        """
        return ActivationType.Activation_Tanh
    
    def forward(self, inputs) -> vectors.ndarray:
        """
        Performs the forward pass for Tanh.

        ### Parameters
        1. inputs (vectors.ndarray): The input values.

        ### Returns
        - vectors.ndarray: Output after applying Tanh.
        """
        super().forward(inputs)
        self.output = vectors.tanh(inputs)
        return self.output

    def backward(self, dvalues) -> None:
        """
        Performs the backward pass for Tanh.

        ### Parameters
        1. dvalues (vectors.ndarray): The gradient of the loss with respect to the inputs.
        """
        self.dinputs = dvalues * (1 - vectors.square(self.output))

    def predictions(self, outputs) -> vectors.ndarray:
        """
        Calculates predictions for Tanh.

        ### Parameters
        1. outputs (vectors.ndarray): The output values.

        ### Returns
        - vectors.ndarray: Predicted values based on the Tanh activation.
        """
        return vectors.tanh(outputs)

# Activation softmax
class Activation_Softmax(Activation):
    """
    Activation function Softmax.

    Inherits from Activation class.

    ### Methods
    1. `__init__(inputs)`: Initializes the class.
    2. `forward(inputs)`: Performs forward pass.
    3. `backward(dvalues)`: Performs backward pass.
    4. `predictions(outputs)`: Calculates predictions.
    """

    def __init__(self) -> None:
        """
        Initializes the Activation Softmax class.
        """
        super().__init__()
    
    def getEnum(self) -> ActivationType:
        """
        Gets the enum type

        ### Returns:
        - ActivationType: Activation type enum
        """
        return ActivationType.Activation_Softmax
    
    def forward(self, inputs) -> vectors.ndarray:
        """
        Performs the forward pass for Softmax.

        ### Parameters
        1. inputs (vectors.ndarray): The input values.

        ### Returns
        - vectors.ndarray: Output after applying Softmax.
        """
        super().forward(inputs)
        exp_values = vectors.exp(inputs - vectors.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / vectors.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output

    def backward(self, dvalues) -> None:
        """
        Performs the backward pass for Softmax.

        ### Parameters
        1. dvalues (vectors.ndarray): The gradient of the loss with respect to the inputs.
        """
        self.dinputs = vectors.empty_like(dvalues)
        for index, (singleOutput, singleDvalues) in enumerate(zip(self.output, dvalues)):
            singleOutput = singleOutput.reshape(-1, 1)
            jacobianMatrix = vectors.diagflat(singleOutput) - vectors.dot(singleOutput, singleOutput.T)
            
            self.dinputs[index] = vectors.dot(jacobianMatrix, singleDvalues)

    def predictions(self, outputs) -> vectors.ndarray:
        """
        Calculates predictions for Softmax.

        ### Parameters
        1. outputs (vectors.ndarray): The output values.

        ### Returns
        - vectors.ndarray: Predicted values based on the Softmax activation.
        """
        return vectors.argmax(outputs, axis=1)

# Combination of loss categorical crossentropy and activation softmax
class Activation_Softmax_Loss_CategoricalCrossentropy:
    """
    Combination of softmax activation and categorical cross-entropy loss.

    ### Methods
    1. `__init__(inputs)`: Initializes the class.
    2. `forward(inputs)`: Performs forward pass.
    3. `backward(dvalues)`: Performs backward pass.
    4. `predictions(outputs)`: Calculates predictions.
    """

    def __init__(self) -> None:
        """
        Initializes the Activation Softmax, with loss categorical class.
        """
        super().__init__()
        self.activation = Activation_Softmax()
        self.loss = Loss.Loss_CategoricalCrossentropy()
    
    def getEnum(self) -> ActivationType:
        """
        Gets the enum type

        ### Returns:
        - ActivationType: Activation type enum
        """
        return ActivationType.Activation_Softmax_Loss_CategoricalCrossentropy
    
    def forward(self, inputs) -> vectors.ndarray:
        """
        Performs the forward pass for the combination.

        ### Parameters
        1. inputs (vectors.ndarray): The input values.

        ### Returns
        - vectors.ndarray: Output after applying the combination of Softmax activation and categorical cross-entropy loss.
        """
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.output

    def backward(self, dvalues, expectedOutputs) -> None:
        """
        Performs the backward pass for the combination.

        ### Parameters
        1. dvalues (vectors.ndarray): The gradient of the loss with respect to the inputs.
        2. expectedOutputs (vectors.ndarray): True labels.
        """
        samples = len(dvalues)
        if len(expectedOutputs.shape) == 2:
            expectedOutputs = vectors.argmax(expectedOutputs, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), expectedOutputs] -= 1
        self.dinputs = self.dinputs / samples

    def predictions(self, outputs) -> vectors.ndarray:
        """
        Calculates predictions for the combination.

        ### Parameters
        1. outputs (vectors.ndarray): The output values.

        ### Returns
        - vectors.ndarray: Predicted values based on the combination.
        """
        return self.activation.predictions(outputs)

activationTypeMapping = {
    ActivationType.Activation_ReLU.value: Activation_ReLU,
    ActivationType.Activation_Softmax.value: Activation_Softmax,
    ActivationType.Activation_Sigmoid.value: Activation_Sigmoid,
    ActivationType.Activation_Tanh.value: Activation_Tanh,
    ActivationType.Activation_Softmax_Loss_CategoricalCrossentropy.value: Activation_Softmax_Loss_CategoricalCrossentropy
}