from Optimization.Algorithm import *
import json, random, math
import NeuralNetwork.Model as Model, NeuralNetwork.Layer as Layer
import time
import copy
from typing import List
import NeuralNetwork.Utils as Utils

# Class Genetic
class Genetic(Algorithm):
	"""
		Represents a genetic algorithm for optimization.

		### Methods
		1. `__init__(populationSize, geneStabilizationFactor, mutationProbability, iterations, crossResults, networksManager)`: Initializes the Genetic algorithm.
		2. `findBestSolution()`: Finds the best solution using the Genetic algorithm.
		3. `_generateInitialPopulation()`: Generates the initial population for the Genetic algorithm.
		4. `mutate(chromosome)`: Mutates the given chromosome.
		5. `mutateModelParameters(chromosome)`: Mutates parameters of the given chromosome's model.
		6. `mutateLayers(firstChromLayers)`: Mutates layers of the given chromosome.
		7. `mutateLayersParameters(firstChromLayers)`: Mutates parameters of the given chromosome's layers.
		8. `cross(firstChrom, secondChrom)`: Crosses two chromosomes and returns their offspring.
		9. `crossLayers(firstChromLayers, secondChromLayers)`: Crosses layers of two chromosomes and returns a list of crossed layers.
		10. `solve()`: Executes the Genetic algorithm to solve the optimization problem.
		"""
	_actualPoblationSolutions : dict

	def __init__(self, populationSize = 5, geneStabilizationFactor = 0.50, mutationProbability = 0.1,
				iterations = 100, crossResults = 2, networksManager = None) -> None:
		"""
		Initializes the Genetic algorithm.

		### Parameters:
		1. populationSize (int): Size of the population. Defaults to 5.
		2. geneStabilizationFactor (float): Gene stabilization factor. Defaults to 0.50.
		3. mutationProbability (float): Probability of mutation. Defaults to 0.1.
		4. iterations (int): Number of iterations. Defaults to 100.
		5. crossResults (int): Number of results from crossover. Defaults to 2.
		6. networksManager: Manager for networks. Defaults to None.
		"""
		super().__init__(networksManager)
		self._actualPoblationSolutions = {}
		# Pass arguments
		self.populationSize = populationSize
		self.mutationProbability = mutationProbability
		self.iterationsNumber = iterations
		self.crossResults = crossResults
		self.geneStabilizationFactor = geneStabilizationFactor

	def findBestSolution(self) -> None:
		"""
		Finds the best solution using the Genetic algorithm.
		"""
		# Clears the list of actual population solutions
		self._actualPoblationSolutions.clear()
		
		# Generates the initial population
		self._generateInitialPopulation()
		
		# Sets the initial time
		self._initTime = time.time()
		
		# Solves the problem
		self.solve()
		
		# Sets the final time
		self._finalTime = time.time()
		
		# Saves the best solution
		self._solutionBest.saveModel("GeneticModelOptimization.gen")

	def _generateInitialPopulation(self):
		"""
		Generates the initial population for the Genetic algorithm.
		"""
		print("Generating initial population...")
		
		for i in range(self.populationSize):
			# Create a random sequential model
			model = self._networksManager.createRandomSequentialModel()
			batchSize = random.choice(self._networksManager.validBatchSizes)
			
			# Check if the model is repeated
			if model not in self._actualPoblationSolutions:
				# Train the model
				model.train(
					inputTrainData=self._networksManager.X_train,
					outputTrainData=self._networksManager.Y_train,
					inputValidationData=self._networksManager.X_validate,
					outputValidationData=self._networksManager.Y_train,
					failureCeiling=1,
					batchSize=batchSize,
					checkEvery=5,
					shuffle=True,
					printProgress=False
				)
				# Add the model to the population
				self._actualPoblationSolutions[model] = (self.evaluateSolution(model), model)
			else:
				i -= 1
		
		# Sort the population
		self._actualPoblationSolutions = dict(sorted(self._actualPoblationSolutions.items(), key=lambda x: x[1][0], reverse=False))
		values = list(self._actualPoblationSolutions.values())
		
		# Assign new Z and best solution
		self._Z_Best = copy.copy(values[0][0])
		self._solutionBest = copy.deepcopy(values[0][1])
		
		# Display a message confirming the creation of the initial population
		print(f"Initial population created! Best loss: {values[0][0]}")

	def mutate(self, chromosome: Model.SequentialModel) -> None:
		"""
		Mutates the given chromosome.

		### Parameters:
		1. chromosome (Model.SequentialModel): Chromosome to be mutated.
		"""
		# Mutate only dense layers and output layers
		self.mutateLayers(chromosome.getDenseLayers())
		self.mutateLayersParameters(chromosome.getDenseLayers(), self._networksManager.validL2regularizerRange, self._networksManager.validL1regularizerRange, self._networksManager.validDropoutRange)
		self.mutateLayersParameters([chromosome.getOutputLayer()], self._networksManager.validOutL2regularizerRange, self._networksManager.validOutL1regularizerRange, (0,0))

		self.mutateModelParameters(chromosome)

	def mutateModelParameters(self, chromosome: Model.SequentialModel) -> None:
		"""
		Mutates model parameters within the given chromosome.

		### Steps:
		1. Generates a new learning rate within specified bounds or keeps the current learning rate.
		2. Generates a new decay value within specified bounds or keeps the current decay value.
		3. Configures the chromosome with the updated parameters.

		### Parameters:
		1. chromosome (Model.SequentialModel): Chromosome to mutate.
		"""
		learnRate = random.triangular(self._networksManager.validLearnRate[0], self._networksManager.validLearnRate[1], self._networksManager.validLearnRate[0]) if random.random() < self.mutationProbability else chromosome.getOptimizer().learningRate
		decay = random.triangular(self._networksManager.validDecayRate[0], self._networksManager.validDecayRate[1], self._networksManager.validDecayRate[0]) if random.random() < self.mutationProbability else chromosome.getOptimizer().decay
		
		if random.random() < self.mutationProbability:
			chromosome.configure(
				loss=self._networksManager.modelsLoss(),
				accuracy=self._networksManager.modelsAccuracy(),
				optimizer=self._networksManager.modelsOptimizer(learningRate=learnRate, decay=decay)
			)
		
	def mutateLayers(self, firstChromLayers : List[Layer.Layer]) -> None:
		"""
		Mutates neurons in the given chromosome's layers.

		### Steps:
		1. Resets neurons in layers based on mutation probability.
		2. Swaps elements within the layers based on mutation probability.

		### Parameters:
		1. firstChromLayers (List[Layer.Layer]): List of chromosome layers to mutate.
		"""
		firstChromLayersSize = len(firstChromLayers)

		# Reset neurons
		for i in range(firstChromLayersSize):
			if random.random() < self.mutationProbability:
				neurons = round(random.triangular(self._networksManager.validNeuronsRange[0], self._networksManager.validNeuronsRange[1], self._networksManager.validNeuronsRange[0]))
				firstChromLayers[i].setNeuronsNumber(neurons)

		iterMax = int(firstChromLayersSize / 2)
		while firstChromLayersSize > 0 and random.random() < self.mutationProbability and iterMax >= 0:
			i = random.randint(0, firstChromLayersSize - 1)
			j = random.randint(0, firstChromLayersSize - 1)
			firstChromLayers[i], firstChromLayers[j] = firstChromLayers[j], firstChromLayers[i]  # Swap the elements
			iterMax -= 1
	
	def mutateLayersParameters(self, firstChromLayers : List[Layer.Layer], validL2regularizerRange: tuple, validL1regularizerRange: tuple, validDropoutRange: tuple) -> None:
		"""
		Mutates parameters within the layers of a chromosome.

		### Steps:
		1. Iterates through each layer in the chromosome's layers.
		2. If the mutation probability is met:
			- Adjusts the L2 regularizer values for weights and biases.
			- Adjusts the L1 regularizer values for weights and biases.
			- Adjusts the dropout values.

		### Parameters:
		1. firstChromLayers (List[Layer.Layer]): Layers within the chromosome to be mutated.
		2. validL2regularizerRange (tuple): Valid range for L2 regularizer.
		3. validL1regularizerRange (tuple): Valid range for L1 regularizer.
		4. validDropoutRange (tuple): Valid range for dropout.

		### Mutation Criteria:
		- Adjusts L2 regularizer for weights and biases if random probability is met.
		- Adjusts L1 regularizer for weights and biases if random probability is met.
		- Adjusts dropout if random probability is met.
		"""
		# Parameters
		for layer in firstChromLayers :
			if random.random() < self.mutationProbability:
				# Get the L2 regularizer
				layer.weightRegularizerL2 =  random.triangular(validL2regularizerRange[0], validL2regularizerRange[1], validL2regularizerRange[0])
				layer.biasRegularizerL2 = layer.weightRegularizerL2

			if random.random() < self.mutationProbability:
				# Get the L1 regularizer
				layer.weightRegularizerL1 = random.triangular(validL1regularizerRange[0], validL1regularizerRange[1], validL1regularizerRange[0])
				layer.biasRegularizerL1 = layer.weightRegularizerL1
				
			if random.random() < self.mutationProbability:
				# Get the dropout
				layer.dropout = random.triangular(validDropoutRange[0], validDropoutRange[1], validDropoutRange[0])
		
	def cross(self, firstChrom : Model.SequentialModel, secondChrom : Model.SequentialModel) -> Model.SequentialModel:
		"""
		Crosses two chromosomes and generates child chromosomes.

		### Steps:
		1. Checks for convolution and pooling layers in both chromosomes for preprocessing.
		2. Determines input size based on preprocessing layers or default if none.
		3. Adds preprocessing layers if present (convolutional and pooling).
		4. Sets input layer and extends with dense and output layers from both chromosomes.
		5. Determines learning rate and decay based on gene stabilization factor.
		6. Creates a new sequential model with the crossed layers, learning rate, and decay.
		7. Mutates the resulting model.

		### Parameters:
		1. firstChrom (Model.SequentialModel): First chromosome for crossing. Must be the one with better results.
		2. secondChrom (Model.SequentialModel): Second chromosome for crossing.

		### Returns:
		- Model.SequentialModel: The newly generated model.
		"""

		layers = []

		hasPreprocessingLayers = False

		# Cross convolution layers (if any)
		convolutionLayers = self.crossLayers(firstChrom.getConvolutionLayers(), secondChrom.getConvolutionLayers())
		poolingLayers = self.crossLayers(firstChrom.getPoolingLayers(), secondChrom.getPoolingLayers())

		if len(convolutionLayers) > 0 or len(poolingLayers) > 0 :
			inputSize = Utils.imageSize[0]

		hasPreprocessingLayers = True
		# Add convolutional and pooling
		layers.extend(convolutionLayers)
		layers.extend(poolingLayers)

		if len(poolingLayers) > 0:
			for layer in poolingLayers:
				inputSize /= layer.getKernelDimensions()[0]
			# Assuing RGB
			inputSize = int(inputSize * inputSize * 3)
		else:
			inputSize = int(self._networksManager.X_train.shape[1])
		
		if hasPreprocessingLayers:
			# Set flatten layer
			layers.append(Layer.FlattenLayer())
		# Set input layer
		layers.append(Layer.InputLayer(neuronsNumber=inputSize))
		# Set dense layers
		layers.extend(self.crossLayers(firstChrom.getDenseLayers(), secondChrom.getDenseLayers()))
		# Set output layer
		layers.extend(self.crossLayers([firstChrom.getOutputLayer()], [secondChrom.getOutputLayer()]))

		learnRate = firstChrom.getOptimizer().currentLearningRate if self.geneStabilizationFactor > random.random() else secondChrom.getOptimizer().currentLearningRate
		decay = firstChrom.getOptimizer().decay if self.geneStabilizationFactor > random.random() else secondChrom.getOptimizer().decay
		# Update the genes stabilizer
		self.geneStabilizationFactor += 0.1 * (1 - self.geneStabilizationFactor)

		model = self._networksManager.createSequentialModel(layers, learnRate, decay)
		
		# Mutate
		self.mutate(model)
		return model
	
	def crossLayers(self, firstChromLayers : List[Layer.Layer], secondChromLayers : List[Layer.Layer]) -> List[Layer.Layer]:
		"""
		Crosses layers from two different chromosomes.

		### Steps:
		1. Determines the length of layers in both chromosomes.
		2. Creates an empty list for the crossed layers.
		3. Determines the maximum and minimum number of layers.
		4. Randomly selects the number of layers to be crossed.
		5. Iterates over the selected number of layers and appends them to the crossed layers list.
		6. Compares the neuron numbers of the layers and ensures the layer with the maximum neuron count is at the top.
		7. Swaps elements to place the layer with the maximum neuron count at the beginning.

		### Parameters:
		1. firstChromLayers (List[Layer.Layer]): Layers from the first chromosome.
		2. secondChromLayers (List[Layer.Layer]): Layers from the second chromosome.

		### Returns:
		- List[Layer.Layer]: List of crossed layers.
		"""
		firstChromLayersLen = len(firstChromLayers)
		secondChromLayersLen = len(secondChromLayers)
		layers : List[Layer.Layer] = []
		layersMax = max(firstChromLayersLen, secondChromLayersLen)
		layersMin = min(firstChromLayersLen, secondChromLayersLen)
		layersSize = random.randint(layersMin, layersMax)
		maxI = 0
		for i in range(layersSize):
			if firstChromLayersLen == 0:
				layers.append(secondChromLayers[i % secondChromLayersLen].copy())
				
			elif secondChromLayersLen == 0:
				layers.append(firstChromLayers[i % firstChromLayersLen].copy())
			
			else:
				layers.append(
				firstChromLayers[i % firstChromLayersLen].copy()
				if self.geneStabilizationFactor > random.random()
				else secondChromLayers[i % secondChromLayersLen].copy()
				)
			
		maxI = i if layers[i].getNeuronsNumber() > layers[maxI].getNeuronsNumber() else maxI
		if len(layers) > 0:
			# Swap the elements. Send the maximun layer to the top
			layers[0], layers[maxI] = layers[maxI], layers[0]  # Swap the elements
		return layers

	# Solve
	def solve(self) -> None:
		"""
		Solve the main problem
		"""
		# No improvement count
		noImprovementCount = 0
		# For number of iterations
		while (noImprovementCount < self.iterationsNumber):
			# Changed best solution
			changedBestSolution = False
			# Get list of values
			solutionsList = list(self._actualPoblationSolutions.values())
			# Get the first best chromosome
			firstBestChrom = solutionsList[0][1]
			# Get the second best chromosome
			secondBestChrom = solutionsList[1][1]
			# Cross best chromosomes
			sonTuple = [self.cross(firstBestChrom, secondBestChrom) for _ in range(self.crossResults)]
			# Set the changed poblation value
			changedPoblation = False
			# For
			for son in sonTuple:
				# Check if son is on the poblation. Do not allow repeated members
				if (son not in self._actualPoblationSolutions):
					batchSize = random.choice(self._networksManager.validBatchSizes)
					# Train the model
					son.train(inputTrainData=self._networksManager.X_train,
								outputTrainData=self._networksManager.Y_train,
								inputValidationData=self._networksManager.X_validate,
								outputValidationData=self._networksManager.Y_train,
								failureCeiling=10,
								batchSize=batchSize,
								checkEvery=5, shuffle=True, printProgress=False)
					# Son solution
					sonZ = self.evaluateSolution(son)
					# Delete the last one
					self._actualPoblationSolutions.popitem()
					# Add the son to the poblation
					self._actualPoblationSolutions[son] = (self.evaluateSolution(son), son)
					# Get the gain
					self._Z_Act = sonZ
					# Set the changed poblation value
					changedPoblation = True
					# Compare to best soution
					if (self._Z_Act < self._Z_Best or self._Z_Best == -1):  # Better solution
						# Assign new Z
						self._Z_Best = copy.copy(self._Z_Act)
						# Assign the best solution
						self._solutionBest = copy.deepcopy(son)
						# Changed the best solution
						changedBestSolution = True
			# Check if best solution changed
			if (changedBestSolution):
				print(f"New best solution found! Total loss: {self.getBestZ()}")
				print(f"At: {time.asctime()}")
				# Reset counter
				noImprovementCount = 0
			else:
				# Increment count
				noImprovementCount += 1
			
			if (changedPoblation):
				# Sort
				self._actualPoblationSolutions = dict(sorted(self._actualPoblationSolutions.items(), key=lambda x: x[1][0], reverse=False))
