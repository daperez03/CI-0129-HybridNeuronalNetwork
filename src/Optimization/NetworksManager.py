from NeuralNetwork.Operations import vectors
import NeuralNetwork.Model as Model
import NeuralNetwork.Activation as Activation
import NeuralNetwork.Utils as Utils
import NeuralNetwork.Layer as Layer
import NeuralNetwork.Loss as Loss
import NeuralNetwork.Accuracy as Accuracy
import NeuralNetwork.Optimizer as Optimizer
import NeuralNetwork.Utils as Utils
import random

class NetworksManager:
	"""
	Manages the creation of neural network models with random layers and configurations.

	### Methods
	1. `__init__(self, X_train: vectors, Y_train: vectors, X_validate: vectors, Y_validate: vectors)`: Initializes the NetworksManager.
	2. `createRandomSequentialModel(self) -> Model.SequentialModel`: Creates a random sequential model.
	3. `createSequentialModel(self, layers: list) -> Model.SequentialModel`: Creates a sequential model based on provided layers.
	4. `setModelsConfiguration(self, loss: Loss.Loss = Loss.Loss_CategoricalCrossentropy, accuracy: Accuracy.Accuracy = Accuracy.Accuracy_Categorical, optimizer: Optimizer.Optimizer = Optimizer.Optimizer_Adam) -> "NetworksManager"`: Sets configurations for models.
	5. `setModelsValidParametersRanges(self, validLearnRateRange: tuple = (0.001, 0.01), validDecayRange: tuple = (0, 5e-2), validBatchSizes: tuple = (32, 64, 128, 256))`: Sets valid parameter ranges for models.
	"""

	def __init__(self, X_train: vectors.ndarray, Y_train: vectors.ndarray, X_validate: vectors.ndarray, Y_validate: vectors.ndarray) -> None:
		"""
		Initializes the NetworksManager.

		### Parameters:
		1. X_train (vectors): Training data.
		2. Y_train (vectors): Training labels.
		3. X_validate (vectors): Validation data.
		4. Y_validate (vectors): Validation labels.
		"""
		# Initialization of the NetworksManager class
		self.X_train: vectors.ndarray = X_train
		self.Y_train: vectors.ndarray = Y_train
		self.X_validate: vectors.ndarray = X_validate
		self.Y_validate: vectors.ndarray = Y_validate
		# Call configuration methods by default
		self.setModelsConfiguration()
		self.setModelsValidParametersRanges()

	def createRandomConvolutionLayers(self) -> tuple:
		"""
		Creates random amount of convolution layers, specified on a configuration parameter.

		### Returns:
		- list: Random amount of convolution layers.
		"""
		pass

	def createRandomPoolingLayers(self) -> tuple:
		"""
		Creates random amount of pooling layers, specified on a configuration parameter.

		### Returns:
		- list: Random amount of pooling layers.
		"""
		layers = []
		# Get the number of pooling layers specified on the range
		poolingLayers = random.randint(self.validPoolingLayersAmount[0], self.validPoolingLayersAmount[1])
		poolingDimensions = random.choice(self.validPoolingDimension)
		outputNumber = Utils.imageSize[0]

		for layer in range(poolingLayers):
			if outputNumber % poolingDimensions != 0:
				raise Exception(f"Incompatible pooling dimensions for input size {outputNumber}")
			else:
				outputNumber = outputNumber / poolingDimensions
			# Choose a random type of pooling
			poolingTypeLayer = random.choice(self.validPoolingTypes)
			layers.append(poolingTypeLayer(poolDimensions=(poolingDimensions,poolingDimensions,1)))

		return layers, outputNumber

	def createRandomDenseLayers(self) -> list:
		"""
		Creates random amount of dense layers, specified on a configuration parameter.

		### Returns:
		- list: Random amount of dense layers.
		"""
		layers = []
		# Get the number of dense layers
		denseLayers = random.randint(self.validDenseLayersAmount[0], self.validDenseLayersAmount[1])

		for layer in range(denseLayers):
			# Get the neurons number
			neurons = int(round(random.triangular(self.validNeuronsRange[0], self.validNeuronsRange[1], self.validNeuronsRange[0])))
			# Get activation
			activationFunction = random.choice(self.validActivationsRange)
			# Get the L2 regularizer
			L2regularizer = random.triangular(self.validL2regularizerRange[0], self.validL2regularizerRange[1], self.validL2regularizerRange[0])
			# Get the L1 regularizer
			L1regularizer = random.triangular(self.validL1regularizerRange[0], self.validL1regularizerRange[1], self.validL1regularizerRange[0])
			# Get the dropout
			dropout = random.triangular(self.validDropoutRange[0], self.validDropoutRange[1], self.validDropoutRange[0])
			# Create the layer
			layers.append(Layer.DenseLayer(neuronsNumber=neurons, activationType=activationFunction, dropoutRate=dropout
											, weightRegularizerL1=L1regularizer, biasRegularizerL1=L1regularizer
											, weightRegularizerL2=L2regularizer, biasRegularizerL2=L2regularizer))

		return layers
	
	def createRandomOutputLayer(self) -> Layer.OutputDenseLayer:
		"""
		Create a random output dense layer.

		### Returns:
		- Layer.OutputDenseLayer: Random output dense layer.
		"""
		# Get the activation
		activationFunction = random.choice(self.validOutActivationsRange)
		# Get the L2 regularizer
		L2regularizer = random.triangular(self.validOutL2regularizerRange[0], self.validOutL2regularizerRange[1], self.validOutL2regularizerRange[0])
		# Get the L1 regularizer
		L1regularizer = random.triangular(self.validOutL1regularizerRange[0], self.validOutL1regularizerRange[1], self.validOutL1regularizerRange[0])
		# Create the layer
		return Layer.OutputDenseLayer(neuronsNumber=self.validOutNeurons, activationType=activationFunction, weightRegularizerL1=L1regularizer, biasRegularizerL1=L1regularizer, weightRegularizerL2=L2regularizer, biasRegularizerL2=L2regularizer)
	
	def createRandomSequentialModel(self) -> Model.SequentialModel:
		"""
		Creates a random sequential model.

		### Returns:
		- Model.SequentialModel: A random sequential model.
		"""
		# Method for creating a random sequential model
		hasPreprocessingLayers = False
		layers = []

		# Check if pooling is required
		if hasattr(self, "validPoolingLayersAmount"):
			hasPreprocessingLayers = True
			poolingLayers, inputDimensions = self.createRandomPoolingLayers()
			layers.extend(poolingLayers)
		
		if hasPreprocessingLayers:
			layers.append(Layer.FlattenLayer())
			inputDimensions = int(inputDimensions * inputDimensions * 3)
		else:
			inputDimensions = int(self.X_train.shape[1])

		layers.append(Layer.InputLayer(neuronsNumber=inputDimensions))
		layers.extend(self.createRandomDenseLayers())
		layers.append(self.createRandomOutputLayer())

		return self.createSequentialModel(layers=layers)

	def createSequentialModel(self, layers: list, learnRate = None, decay = None) -> Model.SequentialModel:
		"""
		Creates a sequential model based on provided layers.

		### Parameters:
		1. layers (list): List of layers for the model.

		### Returns:
		- Model.SequentialModel: A sequential model based on the provided layers.
		"""
		# Method for creating a sequential model
		model: Model.SequentialModel = Model.SequentialModel(layers=layers)
		model.configureInitialWeights()

		if learnRate == None:
			# Get the learn and decay rate
			learnRate = random.triangular(self.validLearnRate[0], self.validLearnRate[1], self.validLearnRate[0])

		if decay == None:
			decay = random.triangular(self.validDecayRate[0], self.validDecayRate[1], self.validDecayRate[0])

		model.configure(loss=self.modelsLoss(),
						optimizer=self.modelsOptimizer(learningRate=learnRate, decay=decay),
						accuracy=self.modelsAccuracy())
		return model

	def setDefaultConfiguration(self) -> "NetworksManager":
		"""
		Sets the default configurations for models creation.

		### Returns:
		- NetworksManager: The updated NetworksManager object.
		"""
		self.setModelsConfiguration()
		self.setModelsValidParametersRanges()
		self.setValidPoolingParametersRanges()
		self.setValidDenseParametersRanges()
		self.setValidOutputParametersRanges()
		return self
	
	def setModelsConfiguration(self, loss: Loss.Loss = Loss.Loss_CategoricalCrossentropy,
								accuracy: Accuracy.Accuracy = Accuracy.Accuracy_Categorical,
								optimizer: Optimizer.Optimizer = Optimizer.Optimizer_Adam) -> "NetworksManager":
		"""
		Sets configurations for models.

		### Parameters:
		1. loss (Loss.Loss): Loss function for models.
		2. accuracy (Accuracy.Accuracy): Accuracy calculation method for models.
		3. optimizer (Optimizer.Optimizer): Optimization algorithm for models.

		### Returns:
		- NetworksManager: The updated NetworksManager object.
		"""
		self.modelsLoss: Loss.Loss = loss
		self.modelsAccuracy: Accuracy.Accuracy = accuracy
		self.modelsOptimizer: Optimizer.Optimizer = optimizer
		return self

	def setModelsValidParametersRanges(self, validLearnRateRange: tuple = (0.001, 0.01),
										validDecayRange: tuple = (0, 5e-2),
										validBatchSizes: tuple = (32, 64, 128, 256)) -> "NetworksManager":
		"""
		Sets valid parameter ranges for models.

		### Parameters:
		1. validLearnRateRange (tuple): Range of valid learning rates.
		2. validDecayRange (tuple): Range of valid decay rates.
		3. validBatchSizes (tuple): Valid batch sizes for training.

		### Returns:
		- NetworksManager: The updated NetworksManager object.
		"""
		self.validLearnRate = validLearnRateRange
		self.validDecayRate = validDecayRange
		self.validBatchSizes = validBatchSizes
		return self

	def setValidDenseParametersRanges(self, validDenseLayersAmount: tuple = (0, 1),
										validActivationsRange: tuple = (Activation.ActivationType.Activation_ReLU,
																		Activation.ActivationType.Activation_Sigmoid,
																		Activation.ActivationType.Activation_Tanh),
										validNeuronsRange: tuple = (32, 1024),
										validDropoutRange: tuple = (0.0, 0.5),
										validL2regularizerRange: tuple = (0.0, 5e-5),
										validL1regularizerRange: tuple = (0.0, 5e-5)) -> "NetworksManager":
		"""
		Sets valid parameter ranges for dense layers.

		### Parameters:
		1. validDenseLayersAmount (tuple): Range of valid dense layers.
		2. validActivationsRange (tuple): Range of valid activation functions.
		3. validNeuronsRange (tuple): Range of valid neurons.
		4. validDropoutRange (tuple): Range of valid dropout rates.
		5. validL2regularizerRange (tuple): Range of valid L2 regularizers.
		6. validL1regularizerRange (tuple): Range of valid L1 regularizers.

		### Returns:
		- NetworksManager: The updated NetworksManager object.
		"""
		self.validDenseLayersAmount = validDenseLayersAmount
		self.validActivationsRange = validActivationsRange
		self.validNeuronsRange = validNeuronsRange
		self.validDropoutRange = validDropoutRange
		self.validL2regularizerRange = validL2regularizerRange
		self.validL1regularizerRange = validL1regularizerRange
		return self

	def setValidOutputParametersRanges(self, validActivationsRange: tuple = (Activation.ActivationType.Activation_Softmax,),
										validNeurons: int = 10,
										validL2regularizerRange: tuple = (0.0, 5e-5),
										validL1regularizerRange: tuple = (0.0, 5e-5)) -> "NetworksManager":
		"""
		Sets valid parameter ranges for output layers.

		### Parameters:
		1. validActivationsRange (tuple): Range of valid activation functions.
		2. validNeurons (int): Valid number of output neurons.
		3. validL2regularizerRange (tuple): Range of valid L2 regularizers.
		4. validL1regularizerRange (tuple): Range of valid L1 regularizers.

		### Returns:
		- NetworksManager: The updated NetworksManager object.
		"""
		self.validOutActivationsRange = validActivationsRange
		self.validOutNeurons = validNeurons
		self.validOutL2regularizerRange = validL2regularizerRange
		self.validOutL1regularizerRange = validL1regularizerRange
		return self

	def setValidPoolingParametersRanges(self, validPoolingLayersAmount: tuple = (3, 3),
										validPoolingDimension: tuple = (2, 2),
										validPoolingTypes: tuple = (Layer.MaxPoolingLayer, Layer.AvgPoolingLayer)) -> "NetworksManager":
		"""
		Sets valid parameter ranges for pooling layers.

		### Parameters:
		1. validPoolingLayersAmount (tuple): Range of valid pooling layers.
		2. validPoolingDimension (tuple): Range of valid pooling dimensions.
		3. validPoolingTypes (tuple): Types of valid pooling layers.

		### Returns:
		- NetworksManager: The updated NetworksManager object.
		"""
		self.validPoolingLayersAmount = validPoolingLayersAmount
		self.validPoolingDimension = validPoolingDimension
		self.validPoolingTypes = validPoolingTypes
		return self

	def setValidConvolutionParametersRanges(self, validConvolutionLayersAmount: tuple = (3, 3)) -> "NetworksManager":
		"""
		Sets valid parameter ranges for convolution layers.

		### Parameters:
		1. validConvolutionLayersAmount (tuple): Range of valid convolution layers.

		### Returns:
		- NetworksManager: The updated NetworksManager object.
		"""
		self.validConvolutionLayersAmount = validConvolutionLayersAmount
		return self