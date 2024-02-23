import Optimization.Genetic as Genetic
import Optimization.NetworksManager as NetworksManager
import DatasetManager

# Read the data
X_train, Y_train = DatasetManager.fruitsData("train", shuffle=True)
X_test, Y_test = DatasetManager.fruitsData("test")

# Use a manager of the data
networksManager = NetworksManager.NetworksManager(X_train=X_train, Y_train=Y_train, X_validate=X_test, Y_validate=Y_test)

networksManager.setModelsConfiguration()
networksManager.setModelsValidParametersRanges()
networksManager.setValidPoolingParametersRanges()
networksManager.setValidDenseParametersRanges()
networksManager.setValidOutputParametersRanges()

geneticAlgorithm = Genetic.Genetic(populationSize=10, geneStabilizationFactor=0.5, mutationProbability=0.1, iterations=70, networksManager=networksManager)
geneticAlgorithm.findBestSolution()

geneticAlgorithm.summary()