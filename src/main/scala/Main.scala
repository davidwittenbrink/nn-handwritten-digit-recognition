
/** A neural network designed to recognize handwritten digits.
  * The learning data is taken from the MNIST dataset.
  */
object Main {

  /** The main functions constructs a network with 784 input neurons, 30 neurons in the only hidden layer
    * and 10 output neurons. The network is then trained using Stochastic Gradient Descent using a batch
    * size of 10, a learning rate of 0.3 and 50 epochs. Finally, the test and training score of the network
    * is printed to the command line and the network configuration is stored as a JSON file in the same format
    * that Carmen Popoviciu uses in her JavaScript NN implementation.
    *
    * @see [[https://github.com/CarmenPopoviciu/neural-net]]
    */
  def main(args: Array[String]): Unit = {

    // Train network on MNIST data and print test/train score as well as a JSON snapshot of the network.
    val nnLayers = List(784,30,10)
    val (weights, biases) = NeuralNetClassifier.generateRandomNetConfig(nnLayers)

    val (trainX, trainY) = MNIST.getTrainingSet
    val (trainedWeights, trainedBiases) = NeuralNetClassifier.sgd(weights, biases, 10, 50, 0.3, trainX, trainY)

    println("done!")
    println("Training score:")
    println(NeuralNetClassifier.calcClassificationScore(trainedWeights, trainedBiases, trainX, trainY))

    println("Test score:")
    val (testX, testY) = MNIST.getTestSet
    println(NeuralNetClassifier.calcClassificationScore(trainedWeights, trainedBiases, testX, testY))
    println(NeuralNetClassifier.getJsonSnapshot(trainedWeights, trainedBiases))

  }
}
