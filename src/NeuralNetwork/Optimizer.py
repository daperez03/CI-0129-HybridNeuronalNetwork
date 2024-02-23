from NeuralNetwork.Operations import vectors
from abc import ABC, abstractclassmethod

class Optimizer(ABC):
    """
    Base optimizer class for handling updates to neural network parameters.

    ### Methods
    1. `__init__(learningRate = 1.0, decay = 0.0)`: Initializes the optimizer with a learning rate and decay factor.
    2. `updateLeaningRate()`: Updates the learning rate with decay.
    3. `updateParameters(layer)`: Abstract method to update a layer's parameters.
    4. `updateIteration()`: Updates the number of iterations for the optimizer.
    """

    def __init__(self, learningRate=1.0, decay=0.0) -> None:
        """
        Initializes the optimizer with a learning rate and decay factor.

        ### Parameters
        1. learningRate (float): The learning rate for the optimizer. Default is 1.0.
        2. decay (float): The decay factor for the learning rate. Default is 0.0.
        """
        self.learningRate = learningRate
        self.currentLearningRate = learningRate
        self.decay = decay
        self.iterations = 0
    
    def updateLeaningRate(self) -> "Optimizer":
        """
        Updates the learning rate with decay.
        """
        if self.decay:
            self.currentLearningRate = self.learningRate * (1.0 / (1.0 + self.decay * self.iterations))
        return self
    
    @abstractclassmethod
    def updateParameters(self, layer) -> None:
        """
        Abstract method to update a layer's parameters.

        ### Parameters
        1. layer: The layer to update.
        """
        raise NotImplementedError("Subclasses must implement updateParameters method")

    def updateIteration(self) -> "Optimizer":
        """
        Updates the number of iterations for the optimizer.
        """
        self.iterations += 1
        return self

class Optimizer_SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    Usually a good option for gradient descent. It considers a single learning rate all the layers.

    ### Methods
    1. `__init__(learningRate = 1.0, decay = 0.0, momentum = 0.0)`: Initializes the SGD optimizer with learning rate, decay, and momentum.
    2. `setInitialMomentum(layer)`: Sets initial momentum for a layer.
    3. `updateParameters(layer)`: Updates a layer's parameters using SGD.
    """

    def __init__(self, learningRate=1.0, decay=0.0, momentum=0.0) -> None:
        """
        Initializes the SGD optimizer.

        ### Parameters
        1. learningRate (float): The learning rate for the optimizer. Default is 1.0.
        2. decay (float): The decay factor for the learning rate. Default is 0.0.
        3. momentum (float): The momentum factor for adjusting gradients. Helps to consider the sign of the previous weights adjust in backpropagation Default is 0.0.
        """
        super().__init__(learningRate, decay)
        self.momentum = momentum

    def setInitialMomentum(self, layer) -> None:
        """
        Sets initial momentum for a layer.

        ### Parameters
        1. layer: The layer for which momentum needs to be initialized.
        """
        layer.weightsMomentum = vectors.zeros_like(layer.getWeights())
        layer.biasesMomentum = vectors.zeros_like(layer.getBiases())
    
    def updateParameters(self, layer) -> None:
        """
        Updates a layer's parameters using SGD.

        ### Parameters
        1. layer: The layer to update.
        """
        if layer.isTrainable():
            # Check if momentum is used for gradient descent
            if self.momentum:
                # Initialize momentum if not already done
                if not hasattr(layer, "weightsMomentum"):
                    self.setInitialMomentum(layer)
                
                # Update weights and biases with momentum
                weightsChange = self.momentum * layer.weightsMomentum - (self.currentLearningRate * layer.dweights)
                biasesChange = self.momentum * layer.biasesMomentum - (self.currentLearningRate * layer.dbiases)
                
                layer.weightsMomentum = weightsChange
                layer.biasesMomentum = biasesChange

            else:
                # Update weights and biases without momentum
                weightsChange = - self.currentLearningRate * layer.dweights
                biasesChange = - self.currentLearningRate * layer.dbiases
                
            layer.updateWeights(weightsChange)
            layer.updateBiases(biasesChange)

class Optimizer_Adagrad(Optimizer):
    """
    AdaGrad optimizer considers an individual learning rate for every single layer.

    ### Methods
    1. `__init__(learingRate = 1.0, decay = 0.0, epsilon = 1e-7)`: Initializes the AdaGrad optimizer with learning rate, decay, and epsilon.
    2. `setInitialCaches(layer)`: Sets the initial caches for a layer.
    3. `updateParameters(layer)`: Updates a layer's parameters using AdaGrad.
    """

    def __init__(self, learningRate=1.0, decay=0.0, epsilon=1e-7) -> None:
        """
        Initializes the AdaGrad optimizer.

        ### Parameters
        1. learningRate (float): The learning rate for the optimizer. Default is 1.0.
        2. decay (float): The decay factor for the learning rate. Default is 0.0.
        3. epsilon (float): The epsilon value to prevent division by zero. Default is 1e-7.
        """
        super().__init__(learningRate, decay)
        self.epsilon = epsilon

    def setInitialCaches(self, layer) -> None:
        """
        Sets the initial caches for a layer.

        ### Parameters
        1. layer: The layer for which caches need to be initialized.
        """
        layer.weightsCache = vectors.zeros_like(layer.getWeights())
        layer.biasesCache = vectors.zeros_like(layer.getBiases())

    def updateParameters(self, layer) -> None:
        """
        Updates a layer's parameters using AdaGrad.

        ### Parameters
        1. layer: The layer to update.
        """
        # If it is the first time using cache
        if not hasattr(layer, 'weightsCache'):
            self.setInitialCaches(layer)
        
        # Update the weights cache as the square of the gradient
        layer.weightsCache += layer.dweights**2
        layer.biasesCache += layer.dbiases**2
        
        weightsChange = -self.currentLearningRate * layer.dweights / (vectors.sqrt(layer.weightsCache) + self.epsilon)
        biasesChange = -self.currentLearningRate * layer.dbiases / (vectors.sqrt(layer.biasesCache) + self.epsilon)
        
        layer.updateWeights(weightsChange)
        layer.updateBiases(biasesChange)

class Optimizer_RMSprop(Optimizer):
    """
    RMSprop optimizer is similar to AdaGrad but uses the average of the cache of the weights instead of calculating the square of the whole cache.

    ### Methods
    1. `__init__(learningRate = 0.001, decay = 0.0, epsilon = 1e-7, rho = 0.9)`: Initializes the RMSprop optimizer.
    2. `setInitialCaches(layer)`: Sets the initial caches for a layer.
    3. `updateParameters(layer)`: Updates a layer's parameters using RMSprop.
    """

    def __init__(self, learningRate=0.001, decay=0.0, epsilon=1e-7, rho=0.9) -> None:
        """
        Initializes the RMSprop optimizer.

        ### Parameters
        1. learningRate (float): The learning rate for the optimizer. Should be small, because it has some huge spikes. Default is 0.001.
        2. decay (float): The decay factor for the learning rate. Default is 0.0.
        3. epsilon (float): The epsilon value to prevent division by zero. Default is 1e-7.
        4. rho (float): The cache memory decay rate hyperparameter. Default is 0.9.
        """
        super().__init__(learningRate, decay)
        self.epsilon = epsilon
        self.rho = rho

    def setInitialCaches(self, layer) -> None:
        """
        Sets the initial caches for a layer.

        ### Parameters
        1. layer: The layer for which caches need to be initialized.
        """
        layer.weightsCache = vectors.zeros_like(layer.getWeights())
        layer.biasCache = vectors.zeros_like(layer.getBiases())
        
    def updateParameters(self, layer) -> None:
        """
        Updates a layer's parameters using RMSprop.

        ### Parameters
        1. layer: The layer to update.
        """
        # If it is the first time using cache
        if not hasattr(layer, 'weightsCache'):
            self.setInitialCaches(layer)
        
        # The greater that rho is, the more importance that cache has (affects more the future caches)
        layer.weightsCache = self.rho * layer.weightsCache + (1 - self.rho) * layer.dweights**2
        layer.biasCache = self.rho * layer.biasCache + (1 - self.rho) * layer.dbiases**2
        
        weightsChange = -self.currentLearningRate * layer.dweights / (vectors.sqrt(layer.weightsCache) + self.epsilon)
        biasesChange = -self.currentLearningRate * layer.dbiases / (vectors.sqrt(layer.biasCache) + self.epsilon)
        
        layer.updateWeights(weightsChange)
        layer.updateBiases(biasesChange)

class Optimizer_Adam(Optimizer):
    """
    Adam optimizer combines the momentum concept with the learning rate per weight.

    ### Methods
    1. `__init__(learningRate=0.001, decay=0.0, epsilon=1e-7, beta1=0.9, beta2=0.999)`: Initializes the Adam optimizer.
    2. `setInitialCachesAndMomentums(layer)`: Sets the initial caches and momentums for a layer.
    3. `updateParameters(layer)`: Updates a layer's parameters using Adam optimizer.
    """

    def __init__(self, learningRate=0.001, decay=0.0, epsilon=1e-7, beta1=0.9, beta2=0.999) -> None:
        """
        Initializes the Adam optimizer.

        ### Parameters
        1. learningRate (float): The learning rate for the optimizer. Default is 0.001.
        2. decay (float): The decay factor for the learning rate. Default is 0.0.
        3. epsilon (float): The epsilon value to prevent division by zero. Default is 1e-7.
        4. beta1 (float): The hyperparameter to modify the momentum, compensating for the initial zero values of the momentums. Default is 0.9.
        5. beta2 (float): The hyperparameter to modify the cache, compensating for the initial zero values of the caches. Default is 0.999.
        """
        super().__init__(learningRate, decay)
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
    
    def setInitialCachesAndMomentums(self, layer) -> None:
        """
        Sets the initial caches and momentums for a layer.

        ### Parameters
        1. layer: The layer for which caches and momentums need to be initialized.
        """
        layer.weightsMomentum = vectors.zeros_like(layer.getWeights())
        layer.weightsCache = vectors.zeros_like(layer.getWeights())
        layer.biasesMomentum = vectors.zeros_like(layer.getBiases())
        layer.biasesCache = vectors.zeros_like(layer.getBiases())

    def updateParameters(self, layer) -> None:
        """
        Updates a layer's parameters using Adam optimizer.

        ### Parameters
        1. layer: The layer to update.
        """
        if layer.isTrainable():
            # If it is the first time using caches or momentum
            if not hasattr(layer, 'weightsMomentum'):
                self.setInitialCachesAndMomentums(layer)
            
            # Calculate the initial momentum by applying beta 1 to the previous momentum
            layer.weightsMomentum = self.beta1 * layer.weightsMomentum + (1 - self.beta1) * layer.dweights
            layer.biasesMomentum = self.beta1 * layer.biasesMomentum + (1 - self.beta1) * layer.dbiases
        
            # Make an adjustment based on beta 1, to compensate for the initial zero value.
            # The more iterations, the lower this adjustment will be
            adjustedWeightsMomentum = layer.weightsMomentum / (1 - self.beta1**(self.iterations + 1))
            adjustedBiasesMomentum = layer.biasesMomentum / (1 - self.beta1**(self.iterations + 1))
            
            # Calculate the initial cache by applying beta 2 to the previous momentum
            layer.weightsCache = self.beta2 * layer.weightsCache + (1 - self.beta2) * layer.dweights**2
            layer.biasesCache = self.beta2 * layer.biasesCache + (1 - self.beta2) * layer.dbiases**2
            
            # Make an adjustment based on beta 2, to compensate for the initial zero value.
            # The more iterations, the lower this adjustment will be
            adjustedWeightsCache = layer.weightsCache / (1 - self.beta2**(self.iterations + 1))
            adjustedBiasesCache = layer.biasesCache / (1 - self.beta2**(self.iterations + 1))
            
            weightsChange = -self.currentLearningRate * adjustedWeightsMomentum / (vectors.sqrt(adjustedWeightsCache) + self.epsilon)
            biasesChange = -self.currentLearningRate * adjustedBiasesMomentum / (vectors.sqrt(adjustedBiasesCache) + self.epsilon)
            
            layer.updateWeights(weightsChange)
            layer.updateBiases(biasesChange)
