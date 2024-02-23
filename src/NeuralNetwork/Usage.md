## Features

### Neural Network Model

The `NeuralNetwork.Model` module provides a `SequentialModel` class, allowing for the creation of neural network models with a sequential architecture.

### Layers

- **MaxPoolingLayer**: Performs max pooling on input data.
- **MinPoolingLayer**: Performs min pooling on input data.
- **AvgPoolingLayer**: Performs avg pooling on input data.
- **FlattenLayer**: Flattens the input data.
- **InputLayer**: Represents the input layer of the neural network.
- **DenseLayer**: Fully connected layer with configurable activation, regularization, and dropout.
- **OutputDenseLayer**: Specialized dense layer for output, often used with softmax activation for classification tasks.

### Activation Functions

The `NeuralNetwork.Activation` module includes various activation functions such as ReLU, Tanh, Sigmoid and Softmax, allowing for nonlinear transformations within the neural network.

### Optimizers

- **Adam Optimizer**: Implements the Adam optimization algorithm, widely used for training neural networks.
- **SGD Optimizer**: Stochastic Gradient Descent optimizer. Usually a good option for gradient descent. It considers a single learning rate all the layers.
- **Adagrad Optimizer**: AdaGrad optimizer considers an individual learning rate for every single layer.
- **RMSprop Optimizer**: RMSprop optimizer is similar to AdaGrad but uses the average of the cache of the weights instead of calculating the square of the whole cache.

### Loss Functions

- **Categorical Crossentropy**: Useful for multi-class classification problems, measuring the difference between predicted and actual class distributions.

### Accuracy Metrics

- **Categorical Accuracy**: Calculates the accuracy of categorical predictions.

### Utilities

The `NeuralNetwork.Utils` module contains various utility functions for neural network operations. These utilities include data shuffling, data augmentation, etc.

## Usage

### Trainning

To create and train the network:

1. **Import Necessary Modules**:
    ```python
    import NeuralNetwork.Model as Model
    import NeuralNetwork.Layer as Layer
    import NeuralNetwork.Activation as Activation
    import NeuralNetwork.Optimizer as Optimizer
    import NeuralNetwork.Loss as Loss
    import NeuralNetwork.Accuracy as Accuracy
    import NeuralNetwork.Convolution as Convolution
    import NeuralNetwork.Pooling as Pooling
    import NeuralNetwork.Utils as Utils
    import DatasetManager # As an example
    import nnfs

    nnfs.init()
    ```

2. **Load Data**:
    ```python
    X_train, Y_train, X_test, Y_test = DatasetManager.myData()
    ```

3. **Create Neural Network Model**:
    ```python
    net = Model.SequentialModel(
        layers=[
          Layer.MaxPoolingLayer(poolDimensions=((2,2,1))),
          Layer.MaxPoolingLayer(poolDimensions=((2,2,1))),
          Layer.MaxPoolingLayer(poolDimensions=((2,2,1))),
          Layer.FlattenLayer(),
          Layer.InputLayer(neuronsNumber=2352),
          Layer.DenseLayer(neuronsNumber=512, activationType=Activation.ActivationType.Activation_ReLU, weightRegularizerL2=5e-6, biasRegularizerL2=5e-6, dropoutRate=0.3),
          Layer.DenseLayer(neuronsNumber=256, activationType=Activation.ActivationType.Activation_ReLU, weightRegularizerL2=5e-6, biasRegularizerL2=5e-6, dropoutRate=0.1),
          Layer.OutputDenseLayer(neuronsNumber=10, activationType=Activation.ActivationType.Activation_Softmax)
      ]
    )
    ```

    Or

    ```python
    net = Model.SequentialModel()
    net.addLayer(Layer.InputLayer(neuronsNumber=200))
    net.addLayer(Layer.DenseLayer(neuronsNumber=128, activationType=Activation.ActivationType.Activation_ReLU))
    net.addLayer(Layer.OutputDenseLayer(neuronsNumber=10, activationType=Activation.ActivationType.Activation_Softmax))
    ```

4. **Configure the Model**:
    ```python
    net.configure(
        loss=Loss.Loss_CategoricalCrossentropy(),
        optimizer=Optimizer.Optimizer_Adam(learningRate=0.001, decay=5e-5),
        accuracy=Accuracy.Accuracy_Categorical()
    )
    ```

5. **Configure Initial Weights**:
    ```python
    net.configureInitialWeights()
    ```

6. **Train the Model**:
    If you want to train the model based on a specified number of epochs, you can do it as follows:

    ```python
    net.train(
        inputTrainData=X_train,
        outputTrainData=Y_train,
        inputValidationData=X_test,
        outputValidationData=Y_test,
        epochs=50, showEvery=10
    )
    ```

    If you want to train the model until a trainning loss treshold is reached, you can do it as follows:

    ```python
    # Train until loss 0.5 or lower is reached
    net.train(
        inputTrainData=X_train,
        outputTrainData=Y_train,
        inputValidationData=X_test,
        outputValidationData=Y_test,
        lossCeiling=0.5, showEvery=10
    )
    ```

    If you want to train the model until a number of fails trying to improve the loss is reached, you can do it as follows:

    ```python
    # Train until the model fails to improve for three consecutive epochs printed
    net.train(
        inputTrainData=X_train,
        outputTrainData=Y_train,
        inputValidationData=X_test,
        outputValidationData=Y_test,
        failureCeiling=3, showEvery=10
    )
    ```

    The terminal will print the epochs on the specified rate and other aditional data:

    ```shell
    Epoch: 0, Loss: 4.2116, Acc: 0.1085, Val loss: 2.2435, Val acc: 0.1137, Learn Rate: 0.003000
    Epoch: 10, Loss: 2.1258, Acc: 0.2021, Val loss: 2.0857, Val acc: 0.1243, Learn Rate: 0.002857
    Epoch: 20, Loss: 1.9762, Acc: 0.2342, Val loss: 1.9146, Val acc: 0.2423, Learn Rate: 0.002727
    Epoch: 30, Loss: 1.8578, Acc: 0.2662, Val loss: 1.8657, Val acc: 0.2637, Learn Rate: 0.002609
    Epoch: 40, Loss: 1.8306, Acc: 0.2694, Val loss: 1.8248, Val acc: 0.3045, Learn Rate: 0.002500
    Epoch: 50, Loss: 1.7666, Acc: 0.2846, Val loss: 1.8074, Val acc: 0.3041, Learn Rate: 0.002400
    ```

7. **Save the Model**:
    ```python
    # Save the model to a file name
    net.saveModel(file="ModelTest")
    ```

### Evaluation

1. **Load the Model**
    ```python
    # Load the model
    net = Model.SequentialModel.loadModel(file="ModelTest")
    ```

2. **Test the model**
    ```python
    net.test(X_Test, Y_Test)
    ```

    ```shell
    This will print the results of the forward pass evaluation:
    Test results:
    Loss: 1.8358, Acc: 0.3400
    ```

3. **Get a summary**
    ```python
    # Print a summary of the epochs
    print(net.summary())
    ```

    This will print the summary of the model:

    ```shell
      Sequential Model Summary
      ╒══════════════════╤═══════════╤════════════════════╤═══════════╤═══════════╤═════════════════════════╤═══════════════════════╕
      │ Layer Type       │ Neurons   │ Activation         │ Kernel    │ Dropout   │ Weight regularizer L2   │ Bias regularizer L2   │
      ╞══════════════════╪═══════════╪════════════════════╪═══════════╪═══════════╪═════════════════════════╪═══════════════════════╡
      │ MaxPoolingLayer  │ -         │ -                  │ (2, 2, 1) │ -         │ -                       │ -                     │
      ├──────────────────┼───────────┼────────────────────┼───────────┼───────────┼─────────────────────────┼───────────────────────┤
      │ MaxPoolingLayer  │ -         │ -                  │ (2, 2, 1) │ -         │ -                       │ -                     │
      ├──────────────────┼───────────┼────────────────────┼───────────┼───────────┼─────────────────────────┼───────────────────────┤
      │ MaxPoolingLayer  │ -         │ -                  │ (2, 2, 1) │ -         │ -                       │ -                     │
      ├──────────────────┼───────────┼────────────────────┼───────────┼───────────┼─────────────────────────┼───────────────────────┤
      │ FlattenLayer     │ -         │ -                  │ -         │ -         │ -                       │ -                     │
      ├──────────────────┼───────────┼────────────────────┼───────────┼───────────┼─────────────────────────┼───────────────────────┤
      │ InputLayer       │ 2352      │ -                  │ -         │ -         │ -                       │ -                     │
      ├──────────────────┼───────────┼────────────────────┼───────────┼───────────┼─────────────────────────┼───────────────────────┤
      │ DenseLayer       │ 512       │ Activation_ReLU    │ -         │ 0.3       │ 5e-06                   │ 5e-06                 │
      ├──────────────────┼───────────┼────────────────────┼───────────┼───────────┼─────────────────────────┼───────────────────────┤
      │ DenseLayer       │ 256       │ Activation_ReLU    │ -         │ 0.1       │ 5e-06                   │ 5e-06                 │
      ├──────────────────┼───────────┼────────────────────┼───────────┼───────────┼─────────────────────────┼───────────────────────┤
      │ OutputDenseLayer │ 10        │ Activation_Softmax │ -         │ -         │ -                       │ -                     │
      ╘══════════════════╧═══════════╧════════════════════╧═══════════╧═══════════╧═════════════════════════╧═══════════════════════╛
      ╒══════════════════════════════╤══════════════════════╤════════════════╤══════════════╤═════════╕
      │ Loss function                │ Accuracy function    │ Optimizer      │   Learn rate │   Decay │
      ╞══════════════════════════════╪══════════════════════╪════════════════╪══════════════╪═════════╡
      │ Loss_CategoricalCrossentropy │ Accuracy_Categorical │ Optimizer_Adam │        0.003 │   0.005 │
      ╘══════════════════════════════╧══════════════════════╧════════════════╧══════════════╧═════════╛
    ```

### Predict

1. **Load the Model**
    ```python
    # Load the model
    net = Model.SequentialModel.loadModel(file="ModelTest")
    ```

2. **Predict the data labels**

    ```python
    # Data should be an array of individual data units. The result will be an array with the predicted labels
    predictions = net.predict(data)
    ```

## Adittional information

For more detailed documentation, refer to individual module documentation and code comments within the source files.
