# Scala handwritten digit recognition

For detailed documentation, visit the project's github site at
https://davidwittenbrink.github.io/nn-handwritten-digit-recognition

A scala neural network implementation used to correctly label
handwritten digits. The network is trained using the MNIST
dataset.

The program offers options to save and load snapshot files of networks.
These snapshot files are in the same format as the one used in
Carmen Popoviciu's [awesome neural network
implementation](https://github.com/CarmenPopoviciu/neural-net). 

Additionally, the program supports adjusting parameters such as
the learning rate, batch size for SGD or the number of hidden
layers. You can see all command line options by running the program
with '--help' as argument:
```
  -p, --ploterrors <file>  If specified, plots a graph of training and testing errors to the given location (in png format). E.g. '~/plot.png'.
  -s, --storesnapshot <file>
                           If specified, stores a snapshot of the trained net to the given location (in JSON format). E.g '/snaps/snapshot.json'.
  -c, --configfromsnapshot <file>
                           If specified, loads the network configuration (weights/bias) from the given snapshot file.
  -h, --hiddenlayers <nr of neurons in 1. hidden layer>,<nr of neurons in 2. hidden layer>
                           If specified, stores a snapshot of the trained network in JSON format to the given location. E.g. 'trainedNetwork.json'.
  -e, --epochs <nr of epochs>
                           The number of epochs for stochastic gradient descent.
  -l, --learningrate <learning rate>
                           The learning rate for gradient descent (decimal).
  -b, --batchsize <batchsize
                           The batchsize for stochastic gradient descent.
  --help                   Prints this usage text.
```
As this is a learning project, the
[NeuralNetClassifier object](/src/main/scala/NeuralNetClassifier.scala)
is written in a functional manner.
If you are interested in using the classifier for other problems,
you can check
[the main function of this project](/src/main/scala/Main.scala)
to see how to create and train a network.
