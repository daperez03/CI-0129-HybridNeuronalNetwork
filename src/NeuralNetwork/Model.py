import NeuralNetwork.Layer as Layer, NeuralNetwork.Optimizer as Optimizer, NeuralNetwork.Loss as Loss, NeuralNetwork.Activation as Activation, NeuralNetwork.Accuracy as Accuracy, NeuralNetwork.Utils as Utils
from NeuralNetwork.Operations import vectors
import pickle
import copy
from tabulate import tabulate
from typing import Tuple
from abc import ABC, abstractclassmethod

class Model:
    """
    Sequential Neural Network Model.

    ### Methods
    1. `__init__(layers: list = [])`: Constructor for the Model class.
    """
    def __init__(self, layers : list = []) -> None:
        """
        Initializes the model.

        ### Parameters:
        1. layers (list): The model's layers. Defaults is []
        """
        self._layers = layers
        self._originalLayers = layers

    def __eq__(self, model: "Model") -> bool:
        """
        Equality method
        """
        return all([layer1 == layer2 for (layer1, layer2) in zip(model._layers, self._layers)])
    
    def __hash__(self) -> int:
        """
        Hasihng method
        """
        return hash(tuple([hash(layer) for layer in self._layers]))
    
    def __str__(self) -> str:
        """
        Class string definition
        """
        return self.summary()
    
    def copy(self) -> "Model":
        """
        Copy the model

        ### Returns:
        - Model: New model
        """
        newLayers = [layer.copy() for layer in self._layers]
        return Model(layers=newLayers)
    
    @abstractclassmethod
    def saveModel(self, file: str) -> "Model":
        """
        Saves the model into a file.
        
        ### Parameters:
        1. file (str): File to dump the model into.
        
        ### Returns:
        - Model: This instance.
        """
        raise NotImplementedError("Subclasses must implement saveModel method")

    @staticmethod
    @abstractclassmethod
    # Load the model
    def loadModel(file: str) -> "Model":
        """
        Loads the model into a file.
        
        ### Parameters:
        1. file (str): File to read the model from.
        
        ### Returns:
        - SequentialModel: This instance.
        """
        raise NotImplementedError("Subclasses must implement loadModel method")

    def getLastLoss(self) -> float:
        """
        Gets the last loss reported by the last trainning

        ### Returns:
        - float: Last reported loss
        """
        return self._lastLoss
    
    def getLayers(self) -> list:
        """
        Gets the layers of the model

        ### Returns:
        - list: Layers of the model
        """
        return self._originalLayers
    
    @abstractclassmethod
    def summary(self) -> str:
        """
        Get a summary of the neural network architecture.

        ### Returns:
        - str: Summary of the model
        """
        raise NotImplementedError("Subclasses must implement loadModel method")

# Neural Network
class SequentialModel(Model):
    """
    Sequential Neural Network Model.

    ### Methods
    1. `__init__(layers: list = [])`: Constructor for the Model class.
    2. `saveParameters(file: str) -> Model`: Save the model parameters to a file using pickle.
    3. `loadParameters(file: str) -> Model`: Load the model parameters from a file.
    4. `saveModel(file) -> Model`: Save the entire model (including structure) to a file.
    5. `loadModel(file: str) -> Model`: Load a model (including structure) from a file.
    6. `configure(loss=Loss.Loss, optimizer=Optimizer.Optimizer, accuracy=Accuracy.Accuracy) -> Model`: Configure the model with loss, optimizer, and accuracy.
    7. `configureInitialWeights() -> Model`: Configure the initial weights for each layer.
    8. `loss(expectedOutputs) -> float`: Calculate the loss of the model.
    9. `accuracy(expectedOutputs) -> float`: Calculate the accuracy of the model.
    10. `validate(inputValidationData, outputValidationData, batchSize) -> Tuple[float, float]`: Validate the model on a validation set.
    11. `train(inputTrainData, outputTrainData, inputValidationData=None, outputValidationData=None, failureCeiling=0, lossCeiling=0.00, epochs=10, batchSize=64, showEvery=100, shuffle=False) -> Model`: Train the model on a dataset.
    12. `test(inputTestData, outputTestData, batchSize=64) -> Model`: Test the model on a test dataset.
    13. `predict(inputPredictData, batchSize=64) -> vectors.ndarray`: Make predictions on new data.
    """

    def __init__(self, layers : list = []) -> None:
        """
        Initializes the model.

        ### Parameters:
        1. layers (list): The model's layers. Defaults is []
        """
        super().__init__(layers)
        self._outputLayer = layers[-1]
        self._lastLoss = None
    
    def _setParameters(self, parameters) -> "SequentialModel":
        """
        Sets the parameters of each layer.

        ### Parameters:
        1. parameters (list): Parameters for each layer.

        ### Returns:
        - SequentialModel: This instance
        """
        for (parameterPair, layer) in zip(parameters, self._layers):
            layer._setParameters(*parameterPair)

        return self

    # Get the parameters
    def _getParameters(self) -> list:
        """
        Gets the parameters of each layer.

        ### Returns:
        - list: Parameters of each layer.
        """
        parameters = []
        for layer in self._layers:
            parameters.append(layer._getParameters())
    
        return parameters

    def getConvolutionLayers(self) -> list:
        """
        Gets the convolution layers of the model
        
        ### Returns:
        - list: Convolution layers
        """
        return [layer for layer in self._originalLayers if isinstance(layer, Layer.ConvolutionLayer)]
    
    def getPoolingLayers(self) -> list:
        """
        Gets the pooling layers of the model
        
        ### Returns:
        - list: Pooling layers
        """
        return [layer for layer in self._originalLayers if issubclass(type(layer), Layer.PoolingLayer)]
    
    def getDenseLayers(self) -> list:
        """
        Gets the dense layers of the model
        
        ### Returns:
        - list: Dense layers
        """
        return [layer for layer in self._originalLayers if isinstance(layer, Layer.DenseLayer) and not isinstance(layer, Layer.OutputDenseLayer)]

    def getOutputLayer(self) -> Layer.Layer:
        """
        Gets the ouput layer of the model

        ### Returns:
        - list: Output layer
        """
        return self._outputLayer
    
    def getOptimizer(self) -> Optimizer.Optimizer:
        """
        Gets the model's optimizer

        ### Returns:
        - Optimizer.Optimizer: Model's optimizer
        """
        return self._optimizer
    
    # Save the parameters
    def saveParameters(self, file: str) -> "SequentialModel":
        """
        Saves the parameters of each layer in a file.

        ### Parameters:
        1. file (str): File to dump the parameters.

        ### Returns:
        - SequentialModel: This instance.
        """
        with open (file, 'wb' ) as f:
            pickle.dump(self._getParameters(), f)
        
        return self
    
    # Load the parameters   
    def loadParameters(self, file: str) -> "SequentialModel":
        """
        Loads the parameters of each layer in a file.

        ### Parameters:
        1. file (str): File to read the parameters from.

        ### Returns:
        - SequentialModel: This instance.
        """
        with open (file, 'rb' ) as f:
            self._setParameters(pickle.load(f))

        return self
    
    # Save the model
    def saveModel(self, file: str) -> "SequentialModel":
        """
        Saves the model into a file.
        
        ### Parameters:
        1. file (str): File to dump the model into.
        
        ### Returns:
        - SequentialModel: This instance.
        """
        # Make a deep copy of the current model instance
        model = copy.deepcopy(self)

        # Reset accumulated values in loss and accuracy objects
        model._loss.clean()
        model._accuracy.clean()

        # For each layer, remove inputs, output, and dinputs properties
        for layer in model._layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        # Open a file in the binary-write mode and save the model
        with open(file, 'wb') as f:
            pickle.dump(model, f)

        return self

    @staticmethod
    # Load the model
    def loadModel(file: str) -> "SequentialModel":
        """
        Loads the model into a file.
        
        ### Parameters:
        1. file (str): File to read the model from.
        
        ### Returns:
        - SequentialModel: This instance.
        """
        # Open a file in the binary-write mode and save the model
        with open(file, 'rb') as f:
            model = pickle.load(f)
            
        return model

    # Add a layer
    def addLayer(self, layer : Layer.Layer) -> "SequentialModel":
        """
        Adds a layer to the model

        ### Parameters:
        1. layer (Layer): Layer to add

        ### Returns:
        - SequentialModel: This instance.
        """
        self._layers.append(layer)
        return self
    
    # Initial configuration
    def configure(self, loss = Loss.Loss, optimizer = Optimizer.Optimizer, accuracy = Accuracy.Accuracy) -> "SequentialModel":
        """
        Configures the model
        
        ### Parameters:
        1. loss (Loss): Loss function
        2. optimizer (Optimizer): Model's optimizer
        3. accuracy (Accuracy): Accuracy function
        
        ### Returns:
        - SequentialModel: This instance.
        """
        self._loss = loss
        self._optimizer = optimizer
        self._accuracy = accuracy
        
        # Determine if Loss_CategoricalCrossentropy and Activation_Softmax are being used
        if isinstance(self._outputLayer.getActivation(), Activation.Activation_Softmax) and isinstance(loss, Loss.Loss_CategoricalCrossentropy):
            self._outputLayer.setActivation(Activation.ActivationType.Activation_Softmax_Loss_CategoricalCrossentropy)
            self._loss = self._outputLayer.getActivation().loss
        elif isinstance(self._outputLayer.getActivation(), Activation.Activation_Softmax_Loss_CategoricalCrossentropy):
            self._loss = self._outputLayer.getActivation().loss

        # Set loss layers
        self._loss.setLayers(self._layers)

        # Determine if could be optimized
        if (self._couldBeOptimized()):
            self._separatePreprocessingLayers()

        return self

    def _couldBeOptimized(self) -> bool:
        """
        Determines if the model can be optimized by separating the pooling and other preprocessing layers from the dense layers.

        ### Returns:
        - bool: Whether or not the model can be optimized.
        """
        self._preprocessingLayers = []

        couldBeOptimized = True
        for layer in self._layers:
            if isinstance(layer, Layer.ConvolutionLayer):
                couldBeOptimized = False
        return couldBeOptimized
        
    def _separatePreprocessingLayers(self) -> None:
        """
        Separates the pooling, flatten and other preprocessing layers from the original layers.
        """
        self._preprocessingLayers = []

        # Your original loop to append layers
        for layer in self._originalLayers:
            if isinstance(layer, (Layer.ConvolutionLayer, Layer.PoolingLayer, Layer.FlattenLayer)):
                self._preprocessingLayers.append(layer)

        # Remove the appended layers from self._layers
        self._layers = [layer for layer in self._layers if layer not in self._preprocessingLayers]


    def _hasPreprocessingLayers(self) -> bool:
        """
        Indicates if the model has separated preprocessing layers.

        ### Returns:
        - bool: Whether or not the model has separated preprocessing layers.
        """
        return len(self._preprocessingLayers) > 0
    
    # Configure initial weights for each layer
    def configureInitialWeights(self) -> "SequentialModel":
        """
        Configures the initial weights for each layer of the model.

        ### Returns:
        - SequentialModel: This instance.
        """
        # Configure the initial weights
        previousLayer = None
        for currentLayer in self._layers:
            if previousLayer is not None:
                currentLayer.setInputsNumber(previousLayer.getNeuronsNumber())
            currentLayer.configureInitialWeights()
    
            # Update the previous_layer variable for the next iteration
            previousLayer = currentLayer

        return self

    def _forwardPreprocessing(self, input) -> vectors.ndarray:
        """
        Forward passes an input to the preprocessing layers in the model, if any

        ### Returns
        - vectors.ndarray: Last preprocessing layer output
        """
        for preprocessLayer in self._preprocessingLayers:
            preprocessLayer.forward(input)
            input = preprocessLayer.output
        
        return preprocessLayer.output

    def _tryPreprocessBatch(self, input, batchSteps, batchSize) -> vectors.ndarray:
        """
        Tries to preprocess a batch, only if the model has preprocessing layers.
        
        ### Parameters:
        1. input: Input data to preprocess.
        2. batchSteps: Number of steps to process the batch.
        3. batchSize: Size of each batch.
        
        ### Returns:
        - vectors.ndarray: Preprocessed results as a numpy array.
        """
        preprocessResults = []

        # If was optimized and separated preprocessing layers
        if self._hasPreprocessingLayers():
            for step in range(batchSteps):
                start = step*batchSize
                end = (step+1)*batchSize
                inputData = input[start:end]
                preprocessResults.append(self._forwardPreprocessing(inputData))
        return vectors.array(preprocessResults)

    # Forward input
    def _forwardInput(self, input) -> None:
        """
        Forward the input through the layers.
        
        ### Parameters:
        1. input: Input data.
        """
        for layer in self._layers:
            layer.forward(input)
            input = layer.output

    # Backward output
    def _backwardOutput(self, expectedResults) -> None:
        """
        Backward pass to update gradients.
        
        ### Parameters:
        1. expectedResults: Expected results for backward propagation.
        """
        self._outputLayer.setExpectedResults(expectedResults)

        output = self._outputLayer.output

        for layer in reversed(self._layers):
            layer.backward(output)
            output = layer.dinputs
            self._optimizer.updateParameters(layer)

    # Get the loss
    def loss(self, expectedOutputs) -> float:
        """
        Calculate loss based on expected outputs.
        
        ### Parameters:
        1. expectedOutputs: Expected outputs for calculating loss.
        
        ### Returns:
        - float: Calculated loss.
        """
        dataLoss,  regularizationLoss = self._loss.calculate(self._outputLayer.output, expectedOutputs)
        # Calculate overall loss
        loss = dataLoss + regularizationLoss
        return loss

    # Get the accuracy
    def accuracy(self, expectedOutputs) -> float:
        """
        Calculate accuracy based on expected outputs.
        
        ### Parameters:
        1. expectedOutputs: Expected outputs for calculating accuracy.
        
        ### Returns:
        - float: Calculated accuracy.
        """
        accuracy = self._accuracy.calculate(self._outputLayer.getActivation().predictions(self._outputLayer.output), expectedOutputs)
        return accuracy
    
    # Print epoch
    def _printEpoch(self, epochNumber, loss, accuracy, validationLoss = None, validationAccuracy = None) -> None:
        """
        Print epoch details during training.
        
        ### Parameters:
        1. epochNumber: Current epoch number.
        2. loss: Loss value.
        3. accuracy: Accuracy value.
        4. validationLoss: Loss value for validation data (optional).
        5. validationAccuracy: Accuracy value for validation data (optional).
        """
        if validationLoss is not None:
            # Print Info
            print("Epoch: {}, Loss: {:.4f}, Acc: {:.4f}, Val loss: {:.4f}, Val acc: {:.4f}, Learn Rate: {:.6f}".format(epochNumber, loss, accuracy, validationLoss, validationAccuracy, self._optimizer.currentLearningRate))
        else:
            # Print Info
            print("Epoch: {}, Loss: {:.4f}, Acc: {:.4f}, Learn Rate: {:.6f}".format(epochNumber, loss, accuracy, self._optimizer.currentLearningRate))

    # Print a test
    def _printTest(self, loss, accuracy) -> None:
        """
        Print test results.
        
        ### Parameters:
        1. loss: Loss value.
        2. accuracy: Accuracy value.
        """
        # Print Info
        print("Test results:\nLoss: {:.4f}, Acc: {:.4f}".format(loss, accuracy))

    # Get the batch division
    def _calculateBatchSteps(self, inputData, batchSize) -> int:
        """
        Calculates the batch steps.
        
        ### Parameters:
        1. inputData: Input data to be divided into batches.
        2. batchSize: Size of each batch.
        
        ### Returns:
        - int: Number of iterations required for the batch division.
        """
        iterations = inputData.shape[0] // batchSize
        if (iterations * inputData.shape[0]) < batchSize:
            iterations += 1
        return iterations

    # Process the input and output data with forward and backward
    def _trainData(self, epochNumber, inputData, outputData, calculateEvery) -> None:
        """
        Process input and output data with forward and backward passes.
        
        ### Parameters:
        1. epochNumber: Current epoch number.
        2. inputData: Input data.
        3. outputData: Output data.
        4. calculateEvery: Frequency to calculate loss and accuracy.
        """
        # Forward
        self._forwardInput(inputData)
        
        if not epochNumber % calculateEvery:
            self._loss.addRegisteredLoss(self.loss(outputData))
            self._accuracy.addRegisteredAccuracy(self.accuracy(outputData))
    
        # Backpropagate
        self._backwardOutput(outputData)

    # Test the data
    def _testData(self, inputData, outputData) -> None:
        """
        Test input and output data with forward pass only.
        
        ### Parameters:
        1. inputData: Input data.
        2. outputData: Output data.
        """
        # Forward
        self._forwardInput(inputData)
        self._loss.addRegisteredLoss(self.loss(outputData))
        self._accuracy.addRegisteredAccuracy(self.accuracy(outputData))

    # Predict the data
    def _predictData(self, inputData) -> vectors.ndarray:
        """
        Predict output based on input data.
        
        ### Parameters:
        1. inputData: Input data.
        
        ### Returns:
        - vectors.ndarray: Predicted output as a numpy array.
        """
        # Forward
        self._forwardInput(inputData)
        return self._outputLayer.output

    def _wrapOnNumpyArry(self, data) -> vectors.ndarray:
        """
        Convert input data to a numpy array.
        
        ### Parameters:
        1. data: Data to be converted.
        
        ### Returns:
        - vectors.ndarray: Converted data as a numpy array.
        """
        return vectors.array(data)
    
    def _trainForLoss(self, inputTrainData, outputTrainData, inputValidationData, outputValidationData, lossCeiling, batchSize, showEvery, shuffle) -> None:
        """
        Train the model based on a loss ceiling criterion.
        
        ### Parameters:
        1. inputTrainData: Training input data.
        2. outputTrainData: Training output data.
        3. inputValidationData: Validation input data.
        4. outputValidationData: Validation output data.
        5. lossCeiling: Threshold for stopping training based on loss.
        6. batchSize: Size of each batch.
        7. showEvery: Frequency to display progress.
        8. shuffle: Flag for shuffling data during training.
        """
        # Divide the batch
        batchSteps = self._calculateBatchSteps(inputTrainData, batchSize)

        loss = None
        epoch = 0
        # Preprocess the batch, if possible
        preprocessedResults = self._tryPreprocessBatch(inputTrainData, batchSteps, batchSize)
    
        while loss is None or loss > lossCeiling:
            # Update the learn rate
            self._optimizer.updateLeaningRate()

            # Process a batch, forward and backwards
            for step in range(batchSteps):
                # Calculate the start and end of the batch
                start = step*batchSize
                end = (step+1)*batchSize

                inputData = inputTrainData[start:end] if len(preprocessedResults) == 0 else preprocessedResults[step]
                outputData = outputTrainData[start:end]

                if shuffle:
                    inputData, outputData = Utils.shuffleData(inputData, outputData)
                
                self._trainData(epoch, inputData, outputData, showEvery)

            if not epoch % showEvery:
                # Calculate loss and accuracy
                loss = self._loss.calculateMean()
                accuracy = self._accuracy.calculateMean()
                # Clean the loss and accuracy
                self._loss.clean()
                self._accuracy.clean()

                if inputValidationData is not None and outputValidationData is not None:
                    # Validate
                    valLoss, valAccuracy = self.validate(inputValidationData, outputValidationData, batchSize)
                    self._lastLoss = valLoss
                    # Clean the loss and accuracy
                    self._loss.clean()
                    self._accuracy.clean()
                    if self._shouldPrint:
                        # Print epoch
                        self._printEpoch(epoch, loss, accuracy, valLoss, valAccuracy)
                else:
                    self._lastLoss = loss
                    if self._shouldPrint:
                        self._printEpoch(epoch, loss, accuracy)
            
            self._optimizer.updateIteration()
            epoch += 1

    def _trainForFailure(self, inputTrainData, outputTrainData, inputValidationData, outputValidationData, failureCeiling, batchSize, showEvery, shuffle) -> None:
        """
        Train the model based on a failure ceiling criterion.
        
        ### Parameters:
        1. inputTrainData: Training input data.
        2. outputTrainData: Training output data.
        3. inputValidationData: Validation input data.
        4. outputValidationData: Validation output data.
        5. failureCeiling: Threshold for stopping training based on consecutive failures.
        6. batchSize: Size of each batch.
        7. showEvery: Frequency to display progress.
        8. shuffle: Flag for shuffling data during training.
        """
        # Divide the batch
        batchSteps = self._calculateBatchSteps(inputTrainData, batchSize)

        # Preprocess the batch, if possible
        preprocessedResults = self._tryPreprocessBatch(inputTrainData, batchSteps, batchSize)

        lastLoss = None
        lastAcc = None
        epoch = 0
        failureCount = 0

        while failureCount < failureCeiling:
            # Update the learn rate
            self._optimizer.updateLeaningRate()

            # Process a batch, forward and backwards
            for step in range(batchSteps):
                # Calculate the start and end of the batch
                start = step*batchSize
                end = (step+1)*batchSize

                inputData = inputTrainData[start:end] if len(preprocessedResults) == 0 else preprocessedResults[step]
                outputData = outputTrainData[start:end]

                if shuffle:
                    inputData, outputData = Utils.shuffleData(inputData, outputData)

                self._trainData(epoch, inputData, outputData, showEvery)

            if not epoch % showEvery:
                # Calculate loss and accuracy
                loss = self._loss.calculateMean()
                accuracy = self._accuracy.calculateMean()
                # Clean the loss and accuracy
                self._loss.clean()
                self._accuracy.clean()

                if inputValidationData is not None and outputValidationData is not None:
                    # Validate
                    valLoss, valAccuracy = self.validate(inputValidationData, outputValidationData, batchSize)
                    currentLoss = valLoss
                    currentAcc = valAccuracy
                    # Clean the loss and accuracy
                    self._loss.clean()
                    self._accuracy.clean()
                    if self._shouldPrint:
                        # Print epoch
                        self._printEpoch(epoch, loss, accuracy, valLoss, valAccuracy)
                else:
                    currentLoss = loss
                    currentAcc = accuracy
                    if self._shouldPrint:
                        self._printEpoch(epoch, loss, accuracy)
    

                if lastLoss is not None:
                    if currentLoss > lastLoss:
                        failureCount += 1
                    elif failureCount != 0:
                        failureCount = 0

                lastLoss = currentLoss
                self._lastLoss = lastLoss


            self._optimizer.updateIteration()
            epoch += 1

    
    def _trainForEpochs(self, inputTrainData, outputTrainData, inputValidationData, outputValidationData, epochs, batchSize, showEvery, shuffle) -> None:
        """
        Train the model for a specified number of epochs.
        
        ### Parameters:
        1. inputTrainData: Training input data.
        2. outputTrainData: Training output data.
        3. inputValidationData: Validation input data.
        4. outputValidationData: Validation output data.
        5. epochs: Number of epochs for training.
        6. batchSize: Size of each batch.
        7. showEvery: Frequency to display progress.
        8. shuffle: Flag for shuffling data during training.
        """
        # Divide the batch
        batchSteps = self._calculateBatchSteps(inputTrainData, batchSize)

        # Preprocess the batch, if possible
        preprocessedResults = self._tryPreprocessBatch(inputTrainData, batchSteps, batchSize)
    
        for epoch in range(epochs+1):
            # Update the learn rate
            self._optimizer.updateLeaningRate()

            # Process a batch, forward and backwards
            for step in range(batchSteps):
                # Calculate the start and end of the batch
                start = step*batchSize
                end = (step+1)*batchSize

                inputData = inputTrainData[start:end] if len(preprocessedResults) == 0 else preprocessedResults[step]
                outputData = outputTrainData[start:end]

                if shuffle:
                    inputData, outputData = Utils.shuffleData(inputData, outputData)
                
                self._trainData(epoch, inputData, outputData, showEvery)

            if not epoch % showEvery:
                # Calculate loss and accuracy
                loss = self._loss.calculateMean()
                accuracy = self._accuracy.calculateMean()
                # Clean the loss and accuracy
                self._loss.clean()
                self._accuracy.clean()

                if inputValidationData is not None and outputValidationData is not None:
                    # Validate
                    valLoss, valAccuracy = self.validate(inputValidationData, outputValidationData, batchSize)
                    self._lastLoss = valLoss
                    # Clean the loss and accuracy
                    self._loss.clean()
                    self._accuracy.clean()
                    if self._shouldPrint:
                        # Print epoch
                        self._printEpoch(epoch, loss, accuracy, valLoss, valAccuracy)
                else:
                    self._lastLoss = loss
                    if self._shouldPrint:
                        self._printEpoch(epoch, loss, accuracy)
            
            self._optimizer.updateIteration()


    def validate(self, inputValidationData, outputValidationData, batchSize) -> Tuple[float, float]:
        """
        Validate the model's performance.
        
        ### Parameters:
        1. inputValidationData: Validation input data.
        2. outputValidationData: Validation output data.
        3. batchSize: Size of each batch.
        
        ### Returns:
        - float, float: Validation loss and accuracy.
        """
        # Divide the batch
        batchSteps = self._calculateBatchSteps(inputValidationData, batchSize)

        # Preprocess the batch, if possible
        preprocessedResults = self._tryPreprocessBatch(inputValidationData, batchSteps, batchSize)
        
        # Process a batch, forward only
        for step in range(batchSteps):
            # Process a batch, forward and backwards
            for step in range(batchSteps):
                # Calculate the start and end of the batch
                start = step*batchSize
                end = (step+1)*batchSize

                inputData = inputValidationData[start:end] if len(preprocessedResults) == 0 else preprocessedResults[step]
                outputData = outputValidationData[start:end]
                self._testData(inputData, outputData)

        return self._loss.calculateMean(), self._accuracy.calculateMean()
    
    # Train
    def train(self, inputTrainData, outputTrainData, inputValidationData = None, outputValidationData = None, failureCeiling = 0, lossCeiling=0.00, epochs = 10, batchSize = 64, checkEvery=10, shuffle=False, printProgress=True) -> "SequentialModel":
        """
        Train the model.
        
        ### Parameters:
        1. inputTrainData: Training input data.
        2. outputTrainData: Training output data.
        3. inputValidationData: Validation input data (optional).
        4. outputValidationData: Validation output data (optional).
        5. failureCeiling: Threshold for stopping training based on consecutive failures (optional).
        6. lossCeiling: Threshold for stopping training based on loss (optional).
        7. epochs: Number of epochs for training. Default is 10.
        8. batchSize: Size of each batch. Default is 64.
        9. checkEvery: Frequency of epochs to display progress or check stopping condition. Default is 10.
        10. shuffle: Flag for shuffling data during training. Default is 10.
        11. printProgress: Defines if the progress should be printed in console
        ### Returns:
        - SequentialModel: This instance.
        """
        self._shouldPrint = printProgress
        # Wrap on numpy arrays
        inputTrainData = self._wrapOnNumpyArry(inputTrainData)
        outputTrainData = self._wrapOnNumpyArry(outputTrainData)

        if inputValidationData is not None and outputValidationData is not None:
            inputValidationData = self._wrapOnNumpyArry(inputValidationData)
            outputValidationData = self._wrapOnNumpyArry(outputValidationData)

        if lossCeiling != 0:
            self._trainForLoss(inputTrainData, outputTrainData, inputValidationData, outputValidationData, lossCeiling, batchSize, checkEvery, shuffle)
        elif failureCeiling != 0:
            self._trainForFailure(inputTrainData, outputTrainData, inputValidationData, outputValidationData, failureCeiling, batchSize, checkEvery, shuffle)
        else:
            self._trainForEpochs(inputTrainData, outputTrainData, inputValidationData, outputValidationData, epochs, batchSize, checkEvery, shuffle)
        
        return self

    # Test
    def test(self, inputTestData, outputTestData, batchSize = 64) -> "SequentialModel":
        """
        Test the model's performance.
        
        ### Parameters:
        1. inputTestData: Test input data.
        2. outputTestData: Test output data.
        3. batchSize: Size of each batch. Default is 64.
        
        ### Returns:
        - SequentialModel: This instance.
        """
        # Wrap on numpy arrays
        inputTestData = self._wrapOnNumpyArry(inputTestData)
        outputTestData = self._wrapOnNumpyArry(outputTestData)
    
        # Divide the batch
        batchSteps = self._calculateBatchSteps(inputTestData, batchSize)

        # Preprocess the batch, if possible
        preprocessedResults = self._tryPreprocessBatch(inputTestData, batchSteps, batchSize)
        
        # Process a batch, forward only
        for step in range(batchSteps):
            # Process a batch, forward and backwards
            for step in range(batchSteps):
                # Calculate the start and end of the batch
                start = step*batchSize
                end = (step+1)*batchSize

                inputData = inputTestData[start:end] if len(preprocessedResults) == 0 else preprocessedResults[step]
                outputData = outputTestData[start:end]
                self._testData(inputData, outputData)
        
        self._printTest(self._loss.calculateMean(), self._accuracy.calculateMean())

        self._loss.clean()
        self._accuracy.clean()
        return self

        
    def predict(self, inputPredictData, batchSize=64) -> vectors.ndarray:
        """
        Make predictions using the model.
        
        ### Parameters:
        1. inputPredictData: Data for prediction.
        2. batchSize: Size of each batch. Default is 64.
        
        ### Returns:
        - vectors.ndarray: Predicted values
        """
        # Wrap on numpy array
        inputPredictData = self._wrapOnNumpyArry(inputPredictData)

        # Divide the batch
        batchSteps = self._calculateBatchSteps(inputPredictData, batchSize)

        collectedOutputs = []

        # Preprocess the batch, if possible
        preprocessedResults = self._tryPreprocessBatch(inputPredictData, batchSteps, batchSize)
        
        # Process a batch, forward only
        for step in range(batchSteps):
            # Calculate the start and end of the batch
            start = step*batchSize
            end = (step+1)*batchSize
            inputData = inputPredictData[start:end] if len(preprocessedResults) == 0 else preprocessedResults[step]
            collectedOutputs.append(self._predictData(inputData))
        
        flattenOutputs = vectors.vstack(collectedOutputs)
        labelsPrediction = self._outputLayer.getActivation().predictions(flattenOutputs)
        return labelsPrediction

    def summary(self) -> str:
        """
        Get a summary of the neural network architecture.

        ### Returns:
        - str: Summary of the model
        """
        BOLD = "\033[1m"
        RESET = "\033[0m"

        headers = [
            f"{BOLD}Layer Type{RESET}",
            f"{BOLD}Neurons{RESET}",
            f"{BOLD}Activation{RESET}",
            f"{BOLD}Kernel{RESET}",
            f"{BOLD}Dropout{RESET}",
            f"{BOLD}Weight regularizer L2{RESET}",
            f"{BOLD}Bias regularizer L2{RESET}"
        ]

        data = []

        for layer in self._originalLayers:
            layer_type = type(layer).__name__
            neurons = getattr(layer, "_neuronsNumber", "-") if getattr(layer, "_neuronsNumber", "-") != 0 else "-"
            function = (type(getattr(layer, "_activation", "-")).__name__ if getattr(layer, "_activation", "-") != "-" else "-")
            function = function if function != "Activation_Softmax_Loss_CategoricalCrossentropy" else "Activation_Softmax"
            function = function if function != "NoneType" else "-"
            kernel = getattr(layer, "_kernelDimensions", "-")
            dropout = getattr(layer, "_dropoutRate", "-") if getattr(layer, "_dropoutRate", "-") != 0 else "-"
            wregularizerL2 = getattr(layer, "weightRegularizerL2", "-") if getattr(layer, "weightRegularizerL2", "-") != 0 else "-"
            bregularizerL2 = getattr(layer, "biasRegularizerL2", "-") if getattr(layer, "biasRegularizerL2", "-") != 0 else "-"
            data.append([layer_type, neurons, function, kernel, dropout, wregularizerL2, bregularizerL2])

        headers2 = [
            f"{BOLD}Loss function{RESET}",
            f"{BOLD}Accuracy function{RESET}",
            f"{BOLD}Optimizer{RESET}",
            f"{BOLD}Learn rate{RESET}",
            f"{BOLD}Decay{RESET}"
        ]

        data2 = []
        data2.append([type(self._loss).__name__,
            type(self._accuracy).__name__,
            type(self._optimizer).__name__,
            str(self._optimizer.learningRate),
            str(self._optimizer.decay)])

        summaryTable = f"\n{BOLD}Sequential Model Summary{RESET}\n"
        summaryTable += tabulate(data, headers=headers, tablefmt='fancy_grid') + "\n"
        summaryTable += tabulate(data2, headers=headers2, tablefmt='fancy_grid')  # Assuming tabulate is imported

        return summaryTable

