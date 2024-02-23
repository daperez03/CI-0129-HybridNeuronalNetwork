import NeuralNetwork.Model as Model, NeuralNetwork.Utils as Utils, DatasetManager

import nnfs

nnfs.init()

image = Utils.readImage(Utils.openDirectoryExplorer(initialDirectory="./Datasets/predict"))
image.show()
image = image.resize((224,224))

net = Model.SequentialModel.loadModel("ModelTest")

predictions = net.predict([image])
print("I think the fruit is a:", DatasetManager.labelsInvertedMapping[predictions[0]])