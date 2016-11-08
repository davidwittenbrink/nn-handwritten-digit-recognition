import java.io.{BufferedWriter, File, FileWriter, IOException}

/** A neural network designed to recognize handwritten digits.
  * The learning data is taken from the MNIST dataset.
  */
object Main {

  /** Writes a string to the given filename
    * @param filename The path of the file to write to
    * @param text The text to write to the file
    */
  def writeStringToFile(filename: String, text: String): Unit = {
    val file = new File(filename)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(text)
    bw.close()
  }

  /** The main functions constructs a network with 784 input neurons, 30 neurons in the only hidden layer
    * and 10 output neurons. The network is then trained using Stochastic Gradient Descent using a batch
    * size of 10, a learning rate of 0.3 and 50 epochs. Finally, the test and training score of the network
    * is printed to the command line. The network configuration can be stored as a JSON file in the same format
    * which Carmen Popoviciu uses in her JavaScript NN implementation.
    *
    * @param args The arguments list. The first argument, if provided, will be used to store a snapshot of the network
    *             to.
    * @see [[https://github.com/CarmenPopoviciu/neural-net]]
    */
  def main(args: Array[String]): Unit = {

    val nnLayers = List(784,30,10)
    val (weights, biases) = NeuralNetClassifier.generateRandomNetConfig(nnLayers)

    val (trainX, trainY) = MNIST.getTrainingSet
    val (trainedWeights, trainedBiases) = NeuralNetClassifier.sgd(weights, biases, 10, 50, 0.3, trainX, trainY)

    println("Training score:")
    println(NeuralNetClassifier.calcClassificationScore(trainedWeights, trainedBiases, trainX, trainY))

    println("Test score:")
    val (testX, testY) = MNIST.getTestSet
    println(NeuralNetClassifier.calcClassificationScore(trainedWeights, trainedBiases, testX, testY))

    if (args.length > 0) {
      try {
        writeStringToFile(args.head, NeuralNetClassifier.getJsonSnapshot(trainedWeights, trainedBiases))
      } catch {
        case ioe: IOException => println("Error writing to File.")
      }
    }
  }
}
