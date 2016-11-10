import java.io.{BufferedWriter, File, FileWriter, IOException}

import breeze.linalg.DenseVector
import breeze.plot._

case class Config(ploterrors: String = null,
                  storesnapshot: String = null,
                  hiddenlayers: Seq[Int] = Seq(30),
                  epochs: Int = 50,
                  learningrate: Double = 0.3,
                  batchsize: Int = 10)


/** A neural network designed to recognize handwritten digits.
  * The learning data is taken from the MNIST dataset.
  */
object Main {

  def savePlots(filename: String, trainingErrors: List[Double], testErrors: List[Double]) = {
    val trainingErrorsVector = DenseVector[Double](trainingErrors.toArray)
    val testErrorsVector = DenseVector[Double](testErrors.toArray)

    val f = Figure()
    val p = f.subplot(0)
    p += plot(DenseVector[Double]((1 to trainingErrorsVector.length).map(_.toDouble).toArray), trainingErrorsVector, name="Training Error")
    p += plot(DenseVector[Double]((1 to testErrorsVector.length).map(_.toDouble).toArray), testErrorsVector, name="Test Error")

    p.legend = true
    p.xlabel = "SGD epochs"
    p.ylabel = "Error"
    f.saveas(filename)
  }

  def parseCommandLineArgs(args: Array[String]): Config = {
    val parser = new scopt.OptionParser[Config]("Neural network for handwritten digit classification") {
      opt[String]('p', "ploterrors").valueName("<file>").
        action( (x, c) => c.copy(ploterrors = x) ).
        text("Plots a graph of training and testing errors to the specified location (in png format).")

      opt[String]('s', "storesnapshot").valueName("<file>").
        action( (x, c) => c.copy(storesnapshot = x) ).
        text("Stores a snapshot of the trained net to the specified location (in JSON format).")

      opt[Seq[Int]]('h', "hiddenlayers").valueName("<nr of neurons in 1. hidden layer>,<nr of neurons in 2. hidden layer>").
        action( (x, c) => c.copy(hiddenlayers = x) ).
        text("Use this to store a snapshot of the trained network in JSON format.")

      opt[Int]('e', "epochs").valueName("<nr of epochs>").action( (x,c) =>
        c.copy(epochs = x) ).text("The number of epochs for stochastic gradient descent")

      opt[Double]('l', "learningrate").valueName("<learning rate>").action( (x,c) =>
        c.copy(learningrate = x) ).text("The learning rate for gradient descent (decimal)")

      opt[Int]('b', "batchsize").valueName("<batchsize").action( (x,c) =>
        c.copy(epochs = x) ).text("The batchsize for stochastic gradient descent")

      help("help").text("prints this usage text")

    }
    parser.parse(args, Config()) match {
      case Some(config) => config
      case None => null
    }
  }

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

    val parsedCommands = parseCommandLineArgs(args)
    if (parsedCommands == null) return

    val nnLayers = 784 +: parsedCommands.hiddenlayers.toList :+ 10
    val (weights, biases) = NeuralNetClassifier.generateRandomNetConfig(nnLayers)

    val (trainX, trainY) = MNIST.getTrainingSet
    val (testX, testY) = MNIST.getTestSet

    val result = NeuralNetClassifier.sgd(weights, biases, parsedCommands.batchsize,
                                         parsedCommands.epochs, parsedCommands.learningrate,
                                         trainX, trainY, testX, testY)
    val (trainedWeights, trainedBiases, trainingErrors, testErrors) = result

    println("Final training error:")
    println(NeuralNetClassifier.calcClassificationError(trainedWeights, trainedBiases, trainX, trainY))

    println("Final test error:")
    println(NeuralNetClassifier.calcClassificationError(trainedWeights, trainedBiases, testX, testY))

    if (parsedCommands.storesnapshot != null) {
      try {
        writeStringToFile(parsedCommands.storesnapshot, NeuralNetClassifier.getJsonSnapshot(trainedWeights, trainedBiases))
      } catch {
        case ioe: IOException => println("Error writing to File.")
      }
    }
    if (parsedCommands.ploterrors != null) {
      savePlots(parsedCommands.ploterrors, trainingErrors, testErrors)
    }
  }
}
