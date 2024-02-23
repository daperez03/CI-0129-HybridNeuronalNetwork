import Optimization.NetworksManager as NetworksManager
from Optimization.NetworksManager import Model
from abc import ABC, abstractclassmethod

# Importance for capacity in Z determination
AMOUNT_IMPORTANCE_FACTOR = 0.01

class Algorithm:
	"""
	Represents an abstract algorithm.

	### Methods
	1. `__init__(networksManager)`: Initializes the Algorithm class.
	2. `setNetworksManager(networksManager)`: Sets the NetworksManager instance for the Algorithm.
	3. `getNetworksManager()`: Retrieves the assigned NetworksManager instance.
	4. `getExecutionTime()`: Calculates the execution time of the Algorithm.
	5. `evaluateSolution(model)`: Evaluates the solution's gain using the last loss of the model.
	6. `findBestSolution()`: Abstract method to find the best solution. Must be implemented by subclasses.
	7. `reset()`: Resets the Algorithm's internal state.
	8. `getBestSolution()`: Retrieves the best solution found.
	9. `getBestZ()`: Retrieves the best Z value found.
	10. `summary()`: Prints a summary of the algorithm's results.
	"""
	_networksManager: NetworksManager.NetworksManager
	_Z_Best: float
	_solutionBest: Model.Model
	_Z_Act: float
	_solutionAct: Model.Model
	_initTime: float
	_finalTime: float

	def __init__(self, networksManager: NetworksManager.NetworksManager) -> None:
		"""
		Initializes the Algorithm class.

		Parameters:
		- networksManager (NetworksManager.NetworksManager): Instance of NetworksManager for the Algorithm.
		"""
		self._networksManager = networksManager
		self._Z_Best = -1
		self._Z_Act = -1
		self._solutionBest = None
		self._solutionAct = None
		self._initTime = 0
		self._finalTime = 0

	def setNetworksManager(self, networksManager: NetworksManager.NetworksManager) -> None:
		"""
		Sets the NetworksManager instance for the Algorithm.

		Parameters:
		- networksManager (NetworksManager.NetworksManager): Instance of NetworksManager.
		"""
		self._networksManager = networksManager

	def getNetworksManager(self) -> NetworksManager.NetworksManager:
		"""
		Retrieves the assigned NetworksManager instance.

		Returns:
		- NetworksManager.NetworksManager: Assigned NetworksManager instance.
		"""
		return self._networksManager

	def getExecutionTime(self) -> float:
		"""
		Calculates the execution time of the Algorithm.

		Returns:
		- float: Execution time.
		"""
		return self._finalTime - self._initTime

	def evaluateSolution(self, model: Model) -> float:
		"""
		Evaluates the solution's gain using the last loss of the model.

		Parameters:
		- model (Model): Model instance.

		Returns:
		- float: Solution gain.
		"""
		return model.getLastLoss()

	@abstractclassmethod
	def findBestSolution(self) -> None:
		"""
		Abstract method to find the best solution. Must be implemented by subclasses.
		"""
		pass

	def reset(self) -> None:
		"""
		Resets the Algorithm's internal state.
		"""
		self._Z_Best = -1
		self._Z_Act = -1
		self._solutionBest.clear() if self._solutionBest else None
		self._solutionAct.clear() if self._solutionAct else None
		self._initTime = 0
		self._finalTime = 0

	def getBestSolution(self) -> list:
		"""
		Retrieves the best solution found.

		Returns:
		- list: Best solution.
		"""
		return self._solutionBest

	def getBestZ(self) -> float:
		"""
		Retrieves the best Z value found.

		Returns:
		- float: Best Z value.
		"""
		return self._Z_Best

	def summary(self) -> None:
		"""
		Prints a summary of the algorithm's results.
		"""
		print('--------ALGORITHM SUMMARY---------')
		print(f'Algorithm name: {self.__class__.__name__}')
		print(f'Best solution found:\n{self.getBestSolution()}')
		print(f'Best Z found: {self.getBestZ()}')
		print(f'Execution time: {self._finalTime - self._initTime}\n')
