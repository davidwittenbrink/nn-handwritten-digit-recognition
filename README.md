# Scala handwritten digit recognition

A scala neural network implementation used to correctly label
handwritten digits. The network is trained using the MNIST
dataset.

As this is a learning project, the
[NeuralNetClassifier object](/src/main/scala/NeuralNetClassifier.scala)
is written in a functional manner.
For an example how the classifier is used, see
[the main function of this project](/src/main/scala/Main.scala).

If the program is executed with a filepath as argument, it will
save a snapshot of the trained network configuration to this
location. The format of the JSON file is the same
Carmen Popoviciu uses in her JavaScript [neural network
implementation](https://github.com/CarmenPopoviciu/neural-net) 
(see her repository for a sample).

The [NeuralNetClassifier object](/src/main/scala/NeuralNetClassifier.scala)
also offers a function to generate a network configuration from a given
JSON snapshot. As an example, a configuration achieving a 77% test score
is given in the [77.json](77.json) file.
It was generated using
* 784 input neurons, 1 hidden layer with 30 neurons, 10 output neurons
* Stochastic gradient descent
  * batch size 10
  * learning rate 0.3
  * 50 epochs
  
For more detailed documentation visit the project's github site at
https://davidwittenbrink.github.io/nn-handwritten-digit-recognition