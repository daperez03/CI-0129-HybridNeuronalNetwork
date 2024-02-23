import NeuralNetwork.Model as Model, DatasetManager

import nnfs

nnfs.init()

X_Test, Y_Test = DatasetManager.fruitsData("test")

net = Model.SequentialModel.loadModel("ModelTest")
net.test(X_Test, Y_Test)
print(net.summary())
