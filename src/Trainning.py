import NeuralNetwork.Model as Model, NeuralNetwork.Layer as Layer, NeuralNetwork.Activation as Activation, NeuralNetwork.Optimizer as Optimizer, NeuralNetwork.Loss as Loss, NeuralNetwork.Accuracy as Accuracy, NeuralNetwork.Convolution as Convolution, NeuralNetwork.Pooling as Pooling, NeuralNetwork.Utils as Utils, DatasetManager
import nnfs

nnfs.init()

X_train, Y_train = DatasetManager.fruitsData("train", shuffle=True)
X_test, Y_test = DatasetManager.fruitsData("test")

# Create the neural network model
net = Model.SequentialModel(
layers=[
    Layer.AvgPoolingLayer(poolDimensions=((2,2,1))),
    Layer.AvgPoolingLayer(poolDimensions=((2,2,1))),
    Layer.AvgPoolingLayer(poolDimensions=((2,2,1))),
    Layer.FlattenLayer(),
    Layer.InputLayer(neuronsNumber=2352),
    Layer.DenseLayer(neuronsNumber=1024, activationType=Activation.ActivationType.Activation_ReLU, weightRegularizerL2=5e-6, biasRegularizerL2=5e-6, dropoutRate=0.3),
    Layer.DenseLayer(neuronsNumber=1024, activationType=Activation.ActivationType.Activation_ReLU, weightRegularizerL2=5e-6, biasRegularizerL2=5e-6, dropoutRate=0.1),
    Layer.OutputDenseLayer(neuronsNumber=10, activationType=Activation.ActivationType.Activation_Softmax)
])

# Configure the neural network
net.configure(
    loss=Loss.Loss_CategoricalCrossentropy(),
    optimizer=Optimizer.Optimizer_Adam(learningRate=0.003, decay=5e-3),
    accuracy=Accuracy.Accuracy_Categorical()
)

# Configure initial weights
net.configureInitialWeights()

# Train the model
net.train(X_train, Y_train, inputValidationData=X_test, outputValidationData=Y_test, epochs=50, checkEvery=10, batchSize=256, shuffle=True)
# Save the model
net.saveModel("ModelTest")
