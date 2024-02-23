import NeuralNetwork.Activation as Activation, NeuralNetwork.Convolution as Convolution, NeuralNetwork.Pooling as Pooling
from NeuralNetwork.Operations import vectors
from abc import ABC, abstractmethod
from typing import Tuple

class Layer(ABC):
    """
    Base class for neural network layers.

    ### Methods
    1. `__init__(neuronsNumber, activationType=None)`: Initializes the layer.
    2. `isTrainable() -> bool`: Checks if the layer is trainable.
    3. `setActivation(activationType) -> Layer`: Sets the activation function for the layer.
    4. `setInputsNumber(numberInputs) -> Layer`: Sets the number of input neurons.
    5. `setExpectedResults(results) -> Layer`: Sets expected results for the layer.
    6. `getWeights() -> vectors.ndarray`: Retrieves the layer's weights.
    7. `getBiases() -> vectors.ndarray`: Retrieves the layer's biases.
    8. `updateWeights(weightsFactor) -> Layer`: Updates the layer's weights.
    9. `updateBiases(biasesFactor) -> Layer`: Updates the layer's biases.
    10. `getNeuronsNumber() -> int`: Retrieves the number of neurons in the layer.
    11. `setParameters(weights, biases) -> Layer`: Sets the parameters (weights and biases) for the layer.
    12. `getParameters() -> Tuple[vectors.ndarray, vectors.ndarray]`: Retrieves the layer's parameters (weights and biases).
    13. `getActivation() -> Activation.Activation`: Retrieves the activation function for the layer.
    14. `configureInitialWeights() -> Layer`: Prepares the initial weights for the layer.
    15. `forward(inputs)`: Performs the forward pass. (Abstract)
    16. `backward(dvalues)`: Performs the backward pass. (Abstract)

    Raises
    - NotImplementedError: Method must be implemented in the subclass.
    """
    def __init__(self, neuronsNumber, activationType: Activation.ActivationType = None) -> None:
        """
        Initializes the layer.
        """
        self._neuronsNumber = neuronsNumber
        if activationType is not None:
            self.setActivation(activationType)
        else:
            self._activation = None
        self._isTrainable = True
        self._weights = vectors.array([])
        self._biases = vectors.array([])
        self._parameters = []  # Typo corrected from _paramaters to _parameters
        self.inputs = vectors.array([])
        self.output = vectors.array([])

    def __eq__(self, layer: "Layer") -> bool:
        """
        Equality method
        """
        return layer.getNeuronsNumber() == self._neuronsNumber and layer.__class__.__name__ == self.__class__.__name__ and layer._activation.__class__.__name__ == self._activation.__class__.__name__

    def __hash__(self) -> int:
        """
        Hasihng method
        """
        return hash((self._neuronsNumber, self.__class__.__name__, self._activation.__class__.__name__))

    @abstractmethod
    def copy(self) -> "Layer":
        """
        Copy the layer

        ### Returns:
        - Layer: New Layer
        """
        raise NotImplementedError("Subclasses must implement copy method")
        
    def isTrainable(self) -> bool:
        """
        Checks if the layer is trainable.

        ### Returns
        - bool: True if the layer is trainable, False otherwise.
        """
        return self._isTrainable

    def setActivation(self, activationType: Activation.ActivationType) -> "Layer":
        """
        Sets the activation function for the layer.

        ### Parameters
        1. activationType (Activation.ActivationType): Type of activation function.

        ### Returns
        - Layer: Updated Layer instance with the set activation.
        """
        self._activation = Activation.activationTypeMapping[activationType.value]()
        return self

    def setNeuronsNumber(self, neuronsNumber) -> "Layer":
        """
        Sets the number of output neurons
        
        ### Parameters:
        1. neuronsNumber (int): Number of output neurons
        
        ### Returns:
        - Layer: Updated Layer instance with the set number of ouputs.
        """
        self._neuronsNumber = neuronsNumber
        return self
    
    def setInputsNumber(self, numberInputs) -> "Layer":
        """
        Sets the number of input neurons.

        ### Parameters
        1. numberInputs (int): Number of input neurons.

        ### Returns
        - Layer: Updated Layer instance with the set number of inputs.
        """
        self._inputsNumber = numberInputs
        return self

    def setExpectedResults(self, results) -> "Layer":
        """
        Sets expected results for the layer.

        ### Parameters
        1. results: Expected results for the layer.

        ### Returns
        - Layer: Updated Layer instance with the set expected results.
        """
        if isinstance(results, list):
            self._expected = vectors.array(results)
        else:
            self._expected = results
        return self

    def getWeights(self) -> vectors.ndarray:
        """Retrieves the layer's weights.
        
        ### Returns
        - vectors.ndarray: The weights
        """
        return self._weights

    def getBiases(self) -> vectors.ndarray:
        """Retrieves the layer's biases.
        
        ### Returns
        - vectors.ndarray: The biases
        """
        return self._biases

    def updateWeights(self, weightsFactor) -> "Layer":
        """
        Updates the layer's weights.

        ### Parameters
        1. weightsFactor: Factor to update the weights by.

        ### Returns
        - Layer: Updated Layer instance with the updated weights.
        """
        self._weights += weightsFactor
        return self

    def updateBiases(self, biasesFactor) -> "Layer":
        """
        Updates the layer's biases.

        ### Parameters
        1. biasesFactor: Factor to update the biases by.

        ### Returns
        - Layer: Updated Layer instance with the updated biases.
        """
        self._biases += biasesFactor
        return self

    def getNeuronsNumber(self) -> int:
        """Retrieves the number of neurons in the layer.
        
        ### Returns
        - int: The number of neurons
        """
        return self._neuronsNumber

    def setParameters(self, weights, biases) -> "Layer":
        """
        Sets the parameters (weights and biases) for the layer.

        ### Parameters
        1. weights: Weights to set.
        2. biases: Biases to set.

        ### Returns
        - Layer: Updated Layer instance with the set parameters.
        """
        self._weights = weights
        self._biases = biases
        return self

    def getParameters(self) -> Tuple[vectors.ndarray, vectors.ndarray]:
        """Retrieves the layer's parameters (weights and biases).
        
        ### Returns
        - vectors.ndarray, vectors.ndarray: The weights and biases
        """
        return self._weights, self._biases

    def getActivation(self) -> Activation.Activation:
        """Retrieves the activation function for the layer.
        
        ### Returns
        - The activation function of the layer
        """
        return self._activation
    
    def configureInitialWeights(self) -> "Layer":
        """Prepares the initial weights for the layer."""
        return self

    @abstractmethod
    def forward(self, inputs) -> None:
        """
        Performs the forward pass.

        ### Parameters
        1. inputs (vectors.ndarray): The input values.

        ### Returns
        - None
        """
        raise NotImplementedError("Method must be implemented in the subclass")

    @abstractmethod
    def backward(self, dvalues) -> None:
        """
        Performs the backward pass.

        ### Parameters
        1. dvalues (vectors.ndarray): The gradient of the loss with respect to the inputs.

        ### Returns
        - None
        """
        raise NotImplementedError("Method must be implemented in the subclass")

class DenseLayer(Layer):
    """
    Dense layer for neural networks.

    ### Methods
    1. `__init__(neuronsNumber, activationType, dropoutRate=0.0, weightRegularizerL1=0, weightRegularizerL2=0, biasRegularizerL1=0, biasRegularizerL2=0)`: Initializes the Dense layer.
    2. `configureInitialWeights()`: Prepares the initial weights for the layer.
    3. `dropoutForward(inputs)`: Forward pass for dropout.
    4. `dropoutBackward(dvalues)`: Backward pass for dropout.
    5. `forward(inputs)`: Performs the forward pass.
    6. `backward(dvalues)`: Performs the backward pass.
    7. `regularize()`: Regularizes the layer's weights and biases.

    Inherits from Layer class.

    ### Attributes
    - `dropoutRate`: Dropout rate for the layer.
    - `weightRegularizerL1`, `weightRegularizerL2`: L1 and L2 regularization strength for weights.
    - `biasRegularizerL1`, `biasRegularizerL2`: L1 and L2 regularization strength for biases.
    """
    def __init__(self, neuronsNumber, activationType: Activation.ActivationType, dropoutRate=0.0,
                 weightRegularizerL1=0, weightRegularizerL2=0, biasRegularizerL1=0, biasRegularizerL2=0) -> None:
        """
        Initializes the Dense layer.

        ### Parameters
        1. neuronsNumber: Number of neurons in the layer.
        2. activationType (Activation.ActivationType): Type of activation function for the layer.
        3. dropoutRate (float): Dropout rate for regularization (default: 0.0).
        4. weightRegularizerL1 (float): L1 regularization strength for weights (default: 0).
        5. weightRegularizerL2 (float): L2 regularization strength for weights (default: 0).
        6. biasRegularizerL1 (float): L1 regularization strength for biases (default: 0).
        7. biasRegularizerL2 (float): L2 regularization strength for biases (default: 0).
        """
        super().__init__(neuronsNumber, activationType)
        self.dropoutRate = dropoutRate
        # Set regularization strength
        self.weightRegularizerL1 = weightRegularizerL1
        self.weightRegularizerL2 = weightRegularizerL2
        self.biasRegularizerL1 = biasRegularizerL1
        self.biasRegularizerL2 = biasRegularizerL2

    def __eq__(self, layer: "Layer") -> bool:
        """
        Equality method
        """
        return super().__eq__(layer) and isinstance(layer, DenseLayer) and layer.dropoutRate == self.dropoutRate and \
                self.weightRegularizerL1 == layer.weightRegularizerL1 and \
                self.weightRegularizerL2 == layer.weightRegularizerL2 and \
                self.biasRegularizerL1 == layer.biasRegularizerL1 and \
                self.biasRegularizerL2 == layer.biasRegularizerL2

    def __hash__(self) -> int:
        """
        Hasihng method
        """
        return hash((super().__hash__(), self.dropoutRate, self.weightRegularizerL1, self.weightRegularizerL2, self.biasRegularizerL1, self.biasRegularizerL2))
        
    def copy(self) -> "DenseLayer":
        """
        Copy the layer

        ### Returns:
        - DenseLayer: New Layer
        """
        return DenseLayer(neuronsNumber=self._neuronsNumber, activationType=self._activation.getEnum()
                          , dropoutRate=self.dropoutRate, weightRegularizerL1=self.weightRegularizerL1, biasRegularizerL1=self.biasRegularizerL1
                          , weightRegularizerL2=self.weightRegularizerL2, biasRegularizerL2=self.biasRegularizerL2)
    
    def configureInitialWeights(self) -> "DenseLayer":
        """
        Prepares the initial weights for the layer.

        ### Returns
        - DenseLayer: Updated DenseLayer instance with initialized weights.
        """
        self._weights = 0.01 * vectors.random.randn(self._inputsNumber, self._neuronsNumber)
        self._biases = vectors.zeros((1, self._neuronsNumber))
        return self

    def dropoutForward(self, inputs) -> None:
        """
        Forward pass for dropout.

        ### Parameters
        1. inputs (vectors.ndarray): The input values.
        """
        self.dropoutRateMask = (vectors.random.rand(*inputs.shape) >= self.dropoutRate) / self.dropoutRate
        self.output = inputs * self.dropoutRateMask
    
    def dropoutBackward(self, dvalues) -> None:
        """
        Backward pass for dropout.

        ### Parameters
        1. dvalues (vectors.ndarray): The gradient of the loss with respect to the inputs.
        """
        self.dinputs = dvalues * self.dropoutRateMask

    def forward(self, inputs) -> None:
        """
        Performs the forward pass.

        ### Parameters
        1. inputs (vectors.ndarray): The input values.
        """
        # Calculate forward
        self.inputs = inputs
        dotProd = vectors.dot(inputs, self._weights) + self._biases
        
        # Calculate forward for activation
        self.output = self._activation.forward(dotProd)

        # Check if dropout
        if self.dropoutRate:
            self.dropoutForward(self.output)

    def backward(self, dvalues) -> None:
        """
        Performs the backward pass.

        ### Parameters
        1. dvalues (vectors.ndarray): The gradient of the loss with respect to the inputs.
        """
        # Check if dropout
        if self.dropoutRate:
            self.dropoutBackward(dvalues)
        else:
            self.dinputs = dvalues
    
        # Calculate backward for activation
        self._activation.backward(self.dinputs)

        # Gradients on parameters
        self.dweights = vectors.dot(self.inputs.T, self._activation.dinputs)
        self.dbiases = vectors.sum(self._activation.dinputs, axis=0, keepdims=True)
        # Regularize if necessary
        self.regularize()
        # Gradient on values
        self.dinputs = vectors.dot(self._activation.dinputs, self._weights.T)

    def regularize(self) -> None:
        """
        Regularizes the layer's weights and biases.
        """
        # L1 on weights
        if self.weightRegularizerL1 > 0:
            dL1 = vectors.ones_like(self._weights)
            dL1[self._weights < 0] = -1
            self.dweights += self.weightRegularizerL1 * dL1

        # L2 on weights
        if self.weightRegularizerL2 > 0:
            self.dweights += 2 * self.weightRegularizerL2 * self._weights

        # L1 on biases
        if self.biasRegularizerL1 > 0:
            dL1 = vectors.ones_like(self._biases)
            dL1[self._biases < 0] = -1
            self.dbiases += self.biasRegularizerL1 * dL1

        # L2 on biases
        if self.biasRegularizerL2 > 0:
            self.dbiases += 2 * self.biasRegularizerL2 * self._biases

class InputLayer(Layer):
    """
    Input layer for neural networks.

    ### Methods
    1. `__init__(neuronsNumber)`: Initializes the Input layer.
    2. `forward(inputs)`: Performs the forward pass.
    3. `backward(dvalues)`: Performs the backward pass.

    Inherits from Layer class.

    ### Attributes
    - `_isTrainable`: Indicates whether the layer is trainable (set as False for InputLayer).
    """
    def __init__(self, neuronsNumber) -> None:
        """
        Initializes the Input layer.

        ### Parameters
        1. neuronsNumber: Number of neurons in the input layer.
        """
        super().__init__(neuronsNumber)
        self._isTrainable = False
    
    def copy(self) -> "InputLayer":
        """
        Copy the layer

        ### Returns:
        - InputLayer: New Layer
        """
        return InputLayer(neuronsNumber=self._neuronsNumber, activationType=self._activation.getEnum()
                          , dropoutRate=self.dropoutRate, weightRegularizerL1=self.weightRegularizerL1, biasRegularizerL1=self.biasRegularizerL1
                          , weightRegularizerL2=self.weightRegularizerL2, biasRegularizerL2=self.biasRegularizerL2)

    def forward(self, inputs) -> None:
        """
        Performs the forward pass.

        ### Parameters
        1. inputs (vectors.ndarray): The input values.
        """
        self.inputs = inputs
        self.output = self.inputs

    def backward(self, dvalues) -> None:
        """
        Performs the backward pass.

        ### Parameters
        1. dvalues (vectors.ndarray): The gradient of the loss with respect to the inputs.
        """
        self.dweights = dvalues
        self.dbiases = dvalues
        self.dinputs = dvalues

class OutputDenseLayer(DenseLayer):
    """
    Output layer for neural networks, inherits from DenseLayer.

    ### Methods
    1. `__init__(neuronsNumber, activationType, dropoutRate=0.0, weightRegularizerL1=0, weightRegularizerL2=0,
                  biasRegularizerL1=0, biasRegularizerL2=0)`: Initializes the OutputDenseLayer.
    2. `forward(inputs)`: Performs the forward pass.
    3. `backward(dvalues)`: Performs the backward pass.

    Inherits from DenseLayer class.
    """
    def __init__(self, neuronsNumber, activationType: Activation.ActivationType, dropoutRate=0.0,
                 weightRegularizerL1=0, weightRegularizerL2=0, biasRegularizerL1=0, biasRegularizerL2=0) -> None:
        """
        Initializes the OutputDenseLayer.

        ### Parameters
        1. neuronsNumber: Number of neurons in the output layer.
        2. activationType (Activation.ActivationType): Type of activation function.
        3. dropoutRate (float): Dropout rate (default = 0.0).
        4. weightRegularizerL1 (float): L1 regularization strength for weights (default = 0).
        5. weightRegularizerL2 (float): L2 regularization strength for weights (default = 0).
        6. biasRegularizerL1 (float): L1 regularization strength for biases (default = 0).
        7. biasRegularizerL2 (float): L2 regularization strength for biases (default = 0).
        """
        super().__init__(neuronsNumber, activationType, dropoutRate)
        # Set regularization strength
        self.weightRegularizerL1 = weightRegularizerL1
        self.weightRegularizerL2 = weightRegularizerL2
        self.biasRegularizerL1 = biasRegularizerL1
        self.biasRegularizerL2 = biasRegularizerL2

    def copy(self) -> "OutputDenseLayer":
        """
        Copy the layer

        ### Returns:
        - OutputDenseLayer: New Layer
        """
        return OutputDenseLayer(neuronsNumber=self._neuronsNumber, activationType=self._activation.getEnum()
                          , dropoutRate=self.dropoutRate, weightRegularizerL1=self.weightRegularizerL1, biasRegularizerL1=self.biasRegularizerL1
                          , weightRegularizerL2=self.weightRegularizerL2, biasRegularizerL2=self.biasRegularizerL2)

    def forward(self, inputs) -> None:
        """
        Performs the forward pass.

        ### Parameters
        1. inputs (vectors.ndarray): The input values.
        """
        # Calculate forward
        self.inputs = inputs
        dotProd = vectors.dot(inputs, self._weights) + self._biases
    
        # Calculate forward for activation
        self.output = self._activation.forward(dotProd)
    
        # Check if dropout
        if self.dropoutRate:
            self.dropoutForward(self.output)

    def backward(self, dvalues) -> None:
        """
        Performs the backward pass.

        ### Parameters
        1. dvalues (vectors.ndarray): The gradient of the loss with respect to the inputs.
        """
        # Check if dropout
        if self.dropoutRate:
            self.dropoutBackward(dvalues)
        else:
            self.dinputs = dvalues
         
        # Calculate backward for activation
        self._activation.backward(self.dinputs, self._expected)

        # Gradients on parameters
        self.dweights = vectors.dot(self.inputs.T, self._activation.dinputs)
        self.dbiases = vectors.sum(self._activation.dinputs, axis=0, keepdims=True)
        # Regularize if necessary
        self.regularize()
        # Gradient on values
        self.dinputs = vectors.dot(self._activation.dinputs, self._weights.T)

class ConvolutionLayer(Layer):
    """
    Convolutional layer for neural networks (currently not fully implemented).

    ### Methods
    1. `__init__(filter)`: Initializes the ConvolutionLayer.
    2. `configureInitialWeights()`: Configures initial weights for the layer.
    3. `forward(image)`: Performs the forward pass.
    4. `backward(dvalues)`: Performs the backward pass.

    Inherits from Layer class.
    """
    def __init__(self, filter: Convolution.FilterType, filtersNumber = 0, kernelSize = 3) -> None:
        """
        Initializes the ConvolutionLayer.

        ### Parameters
        1. filter (Convolution.FilterType): Type of convolutional filter.
        """
        super().__init__(0, None)
        self._kernelSize = kernelSize
        self._filtersNumber = filtersNumber
        self._isTrainable = False
        self._filter = Convolution.filterTypeMapping[filter]

    def copy(self) -> "ConvolutionLayer":
        """
        Copy the layer

        ### Returns:
        - ConvolutionLayer: New Layer
        """
        pass

    def __eq__(self, layer: "Layer") -> bool:
        """
        Equality method
        """
        return super().__eq__(layer) and isinstance(layer, ConvolutionLayer) and layer._kernelSize == self._kernelSize and layer._filtersNumber == self._filtersNumber

    def __hash__(self) -> int:
        """
        Hasihng method
        """
        return hash((super().__hash__(), self._kernelSize, self._filtersNumber))

    def configureInitialWeights(self) -> None:
        """Configures initial weights for the layer."""
        pass

    def forward(self, image) -> None:
        """
        Performs the forward pass.

        ### Parameters
        1. image: Input image data.
        """
        self.inputs = image
        self.output = Convolution.applyConvolutionEfficient(image, self._filter)

    def backward(self, dvalues) -> None:
        """
        Performs the backward pass.

        ### Parameters
        1. dvalues: The gradient of the loss with respect to the inputs.
        """
        self.dweights = dvalues
        self.dbiases = dvalues
        self.dinputs = dvalues


class PoolingLayer(Layer):
    """
    Pooling layer for neural networks.

    ### Methods
    1. `__init__(poolType, poolDimensions)`: Initializes the PoolingLayer.
    2. `configureInitialWeights()`: Configures initial weights for the layer.
    3. `forward(image)`: Performs the forward pass.
    4. `backward(dvalues)`: Performs the backward pass.

    Inherits from Layer class.
    """
    def __init__(self, poolType: Pooling.PoolType, poolDimensions) -> None:
        """
        Initializes the PoolingLayer.

        ### Parameters
        1. poolType (Pooling.PoolType): Type of pooling operation.
        2. poolDimensions: Dimensions of the pooling window.
        """
        super().__init__(0, None)
        self._isTrainable = False
        self._poolType = poolType
        self._kernelDimensions = poolDimensions
    
    def __eq__(self, layer: "Layer") -> bool:
        """
        Equality method
        """
        return super().__eq__(layer) and isinstance(layer, PoolingLayer) and layer._poolType == self._poolType and layer._kernelDimensions == self._kernelDimensions

    def __hash__(self) -> int:
        """
        Hasihng method
        """
        return hash((super().__hash__(), self._poolType, self._kernelDimensions))

    def copy(self) -> "PoolingLayer":
        """
        Copy the layer

        ### Returns:
        - PoolingLayer: New Layer
        """
        return PoolingLayer(poolType=self._poolType, poolDimensions=self._kernelDimensions)
    
    def getKernelDimensions(self) -> tuple:
        """
        Gets the kernel dimensions

        ### Returns:
        - tuple: The kernel dimensions
        """
        return self._kernelDimensions

    def configureInitialWeights(self) -> None:
        """Configures initial weights for the layer."""
        pass

    def forward(self, image) -> None:
        """
        Performs the forward pass.

        ### Parameters
        1. image: Input image data.
        """
        self.inputs = image
        self.output = vectors.array(Pooling.applyPoolingEfficient(image, self._poolType, self._kernelDimensions))

    def backward(self, dvalues) -> None:
        """
        Performs the backward pass.

        ### Parameters
        1. dvalues: The gradient of the loss with respect to the inputs.
        """
        self.dweights = dvalues
        self.dbiases = dvalues
        self.dinputs = dvalues

class MaxPoolingLayer(PoolingLayer):
    """
    Max pooling layer for neural networks.

    ### Methods
    1. `__init__(poolType, poolDimensions)`: Initializes the PoolingLayer.
    2. `configureInitialWeights()`: Configures initial weights for the layer.
    3. `forward(image)`: Performs the forward pass.
    4. `backward(dvalues)`: Performs the backward pass.

    Inherits from Layer class.
    """
    def __init__(self, poolDimensions) -> None:
        """
        Initializes the PoolingLayer.

        ### Parameters
        1. poolDimensions: Dimensions of the pooling window.
        """
        super().__init__(Pooling.PoolType.MAX, poolDimensions)
    
    def copy(self) -> "MaxPoolingLayer":
        """
        Copy the layer

        ### Returns:
        - MaxPoolingLayer: New Layer
        """
        return MaxPoolingLayer(poolDimensions=self._kernelDimensions)
    
    def configureInitialWeights(self) -> None:
        """Configures initial weights for the layer."""
        pass

    def forward(self, image) -> None:
        """
        Performs the forward pass.

        ### Parameters
        1. image: Input image data.
        """
        self.inputs = image
        self.output = vectors.array(Pooling.applyPoolingEfficient(image, self._poolType, self._kernelDimensions))

    def backward(self, dvalues) -> None:
        """
        Performs the backward pass.

        ### Parameters
        1. dvalues: The gradient of the loss with respect to the inputs.
        """
        self.dweights = dvalues
        self.dbiases = dvalues
        self.dinputs = dvalues

class AvgPoolingLayer(PoolingLayer):
    """
    Average pooling layer for neural networks.

    ### Methods
    1. `__init__(poolType, poolDimensions)`: Initializes the PoolingLayer.
    2. `configureInitialWeights()`: Configures initial weights for the layer.
    3. `forward(image)`: Performs the forward pass.
    4. `backward(dvalues)`: Performs the backward pass.

    Inherits from Layer class.
    """
    def __init__(self, poolDimensions) -> None:
        """
        Initializes the PoolingLayer.

        ### Parameters
        1. poolDimensions: Dimensions of the pooling window.
        """
        super().__init__(Pooling.PoolType.AVG, poolDimensions)
    
    def copy(self) -> "AvgPoolingLayer":
        """
        Copy the layer

        ### Returns:
        - AvgPoolingLayer: New Layer
        """
        return AvgPoolingLayer(poolDimensions=self._kernelDimensions)
    
    def configureInitialWeights(self) -> None:
        """Configures initial weights for the layer."""
        pass

    def forward(self, image) -> None:
        """
        Performs the forward pass.

        ### Parameters
        1. image: Input image data.
        """
        self.inputs = image
        self.output = vectors.array(Pooling.applyPoolingEfficient(image, self._poolType, self._kernelDimensions))

    def backward(self, dvalues) -> None:
        """
        Performs the backward pass.

        ### Parameters
        1. dvalues: The gradient of the loss with respect to the inputs.
        """
        self.dweights = dvalues
        self.dbiases = dvalues
        self.dinputs = dvalues

class MinPoolingLayer(PoolingLayer):
    """
    Min pooling layer for neural networks.

    ### Methods
    1. `__init__(poolType, poolDimensions)`: Initializes the PoolingLayer.
    2. `configureInitialWeights()`: Configures initial weights for the layer.
    3. `forward(image)`: Performs the forward pass.
    4. `backward(dvalues)`: Performs the backward pass.

    Inherits from Layer class.
    """
    def __init__(self, poolDimensions) -> None:
        """
        Initializes the PoolingLayer.

        ### Parameters
        1. poolDimensions: Dimensions of the pooling window.
        """
        super().__init__(Pooling.PoolType.MIN, poolDimensions)
    
    def copy(self) -> "MinPoolingLayer":
        """
        Copy the layer

        ### Returns:
        - MinPoolingLayer: New Layer
        """
        return MinPoolingLayer(poolDimensions=self._kernelDimensions)
    
    def configureInitialWeights(self) -> None:
        """Configures initial weights for the layer."""
        pass

    def forward(self, image) -> None:
        """
        Performs the forward pass.

        ### Parameters
        1. image: Input image data.
        """
        self.inputs = image
        self.output = vectors.array(Pooling.applyPoolingEfficient(image, self._poolType, self._kernelDimensions))

    def backward(self, dvalues) -> None:
        """
        Performs the backward pass.

        ### Parameters
        1. dvalues: The gradient of the loss with respect to the inputs.
        """
        self.dweights = dvalues
        self.dbiases = dvalues
        self.dinputs = dvalues
    
class FlattenLayer(Layer):
    """
    Flatten layer for neural networks.

    ### Methods
    1. `__init__()`: Initializes the FlattenLayer.
    2. `configureInitialWeights()`: Configures initial weights for the layer.
    3. `forward(inputs)`: Performs the forward pass.
    4. `backward(dvalues)`: Performs the backward pass.

    Inherits from Layer class.
    """
    def __init__(self) -> None:
        """
        Initializes the FlattenLayer.
        """
        super().__init__(0, None)
        self._isTrainable = False
    
    def __eq__(self, layer: "Layer") -> bool:
        """
        Equality method
        """
        return super().__eq__(layer) and isinstance(layer, FlattenLayer)

    def __hash__(self) -> int:
        """
        Hasihng method
        """
        return super().__hash__()
    
    def copy(self) -> "FlattenLayer":
        """
        Copy the layer

        ### Returns:
        - FlattenLayer: New Layer
        """
        return FlattenLayer()
    
    def configureInitialWeights(self) -> None:
        """Configures initial weights for the layer."""
        pass

    def forward(self, inputs) -> None:
        """
        Performs the forward pass.

        ### Parameters
        1. inputs: Input data.

        ### Note
        Flattens the input data and normalizes it to a range between -1 and 1.
        """
        self.output = []
        self.inputs = inputs
        for input in inputs:
            flattened_input = (vectors.array(input).flatten() / 255)
            self.output.append(flattened_input)

    def backward(self, dvalues) -> None:
        """
        Performs the backward pass.

        ### Parameters
        1. dvalues: The gradient of the loss with respect to the inputs.
        """
        self.dweights = dvalues
        self.dbiases = dvalues
        self.dinputs = dvalues
