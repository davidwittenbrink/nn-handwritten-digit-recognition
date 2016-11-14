import java.io.{BufferedWriter, File, FileWriter, IOException}

import breeze.linalg.DenseVector
import breeze.plot._


/** A class holding the parsed command line arguments. It is populated by the scopt command line
  * option parser.
  *
  * @param ploterrors The destination where the error plot will be stored. E.g. "/plots/errors.png"
  * @param storesnapshot The destination where the created JSON snapshot will be stored. E.g. "~/snap.json"
  * @param configfromsnapshot The file path to the JSON snapshot that will be parsed as the starting weights/bias
  *                           config for the network.
  * @param hiddenlayers A sequence which specifies the number of neurons per hidden layer. E.g. Seq(30, 20) would
  *                     result in a network with two hidden layers, one with 30 and one with 20 neurons.
  * @param epochs The number of epochs to run SGD
  * @param learningrate The learning rate for SGD
  * @param batchsize The batchsize for SGD
  * @see https://github.com/scopt/scopt
  */
case class CliArgs(ploterrors: String = null,
                   storesnapshot: String = null,
                   configfromsnapshot: String = null,
                   hiddenlayers: Seq[Int] = Seq(30),
                   epochs: Int = 50,
                   learningrate: Double = 0.3,
                   batchsize: Int = 10)


/** An implementation of a neural network to recognize handwritten digits.
  * The learning data is taken from the MNIST dataset.
  *
  */
object Main {

  /** Return a network configuration either parsed from a snapshot file or created randomly.
    *
    * @param parsedCliArgs The cli arguments parsed by scopt.
    *                      If parsedCliArgs.configFromSnapshot is null, no file was specified
    * @return A tuple of a list of weights and a list of biases.
    *         If config file was specified, it will be parsed and returned. If not, a random network config is returned.
    * */
  def loadNetConfigRandomlyOrFromFile(parsedCliArgs: CliArgs) = {
    val nnLayers = 784 +: parsedCliArgs.hiddenlayers.toList :+ 10

    if (parsedCliArgs.configfromsnapshot == null) {
      NeuralNetClassifier.generateRandomNetConfig(nnLayers)
    }
    else {
      try {
        val source = io.Source.fromFile(parsedCliArgs.configfromsnapshot)
        val lines = try source.getLines mkString "\n" finally source.close()
        source.close()
        NeuralNetClassifier.getConfigFromSnapshot(lines)
      } catch {
        case e: IOException => {
          println("Error loading file. Using random config.")
          NeuralNetClassifier.generateRandomNetConfig(nnLayers)
        }
      }
    }
  }

  /** Creates and saves a plot of the training and test errors at the position specified via
    * the CLI arguments.
    *
    * @param cliArgs The parsed CLI arguments.
    * @param trainingErrors The list of training errors
    * @param testErrors The list of test errors.
    * */
  def savePlots(cliArgs: CliArgs, trainingErrors: List[Double], testErrors: List[Double]) = {
    val trainingErrorsVector = DenseVector[Double](trainingErrors.toArray)
    val testErrorsVector = DenseVector[Double](testErrors.toArray)

    val f = Figure()
    val p = f.subplot(0)
    p += plot(DenseVector[Double]((1 to trainingErrorsVector.length).map(_.toDouble).toArray), trainingErrorsVector, name="Training Error")
    p += plot(DenseVector[Double]((1 to testErrorsVector.length).map(_.toDouble).toArray), testErrorsVector, name="Test Error")

    p.title = s"Test err=${testErrors.last.toString.substring(0, 4)} | " +
              s"Training err=${trainingErrors.last.toString.substring(0, 4)} | " +
              s"${cliArgs.epochs} epochs |" +
              s"Learning rate=${cliArgs.learningrate} | Batch size=${cliArgs.batchsize} | " +
              s"${cliArgs.hiddenlayers.length} hidden layers"
    p.legend = true
    p.xlabel = "SGD epochs"
    p.ylabel = "Error"
    f.saveas(cliArgs.ploterrors)
  }


  def parseCommandLineArgs(args: Array[String]): CliArgs = {
    val parser = new scopt.OptionParser[CliArgs]("Neural network for handwritten digit classification") {
      opt[String]('p', "ploterrors").valueName("<file>").
        action( (x, c) => c.copy(ploterrors = x) ).
        text("If specified, plots a graph of training and testing errors to the given " +
             "location (in png format). E.g. '~/plot.png'.")

      opt[String]('s', "storesnapshot").valueName("<file>").
        action( (x, c) => c.copy(storesnapshot = x) ).
        text("If specified, stores a snapshot of the trained net to the given location (in JSON format). " +
             "E.g '/snaps/snapshot.json'.")

      opt[String]('c', "configfromsnapshot").valueName("<file>").
        action( (x, c) => c.copy(configfromsnapshot = x) ).
        text("If specified, loads the network configuration (weights/bias) from the given snapshot file.")

      opt[Seq[Int]]('h', "hiddenlayers").valueName("<nr of neurons in 1. hidden layer>,<nr of neurons in 2. hidden layer>").
        action( (x, c) => c.copy(hiddenlayers = x) ).
        text("If specified, stores a snapshot of the trained network in JSON format to the given location. E.g. 'trainedNetwork.json'.")

      opt[Int]('e', "epochs").valueName("<nr of epochs>").action( (x,c) =>
        c.copy(epochs = x) ).text("The number of epochs for stochastic gradient descent.")

      opt[Double]('l', "learningrate").valueName("<learning rate>").action( (x,c) =>
        c.copy(learningrate = x) ).text("The learning rate for gradient descent (decimal).")

      opt[Int]('b', "batchsize").valueName("<batchsize").action( (x,c) =>
        c.copy(batchsize = x) ).text("The batchsize for stochastic gradient descent.")

      help("help").text("Prints this usage text.")

    }
    parser.parse(args, CliArgs()) match {
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
    * is printed to the command line.
    *
    * Multiple command line options allow to adjust this configuration and also to store the trained network to disk
    * or load a previously trained network. For a detailed list of options, run the program with "--help".
    *
    * @param args The arguments list. For a detailed list of options, run the program with "--help"
    * @see [[https://github.com/CarmenPopoviciu/neural-net]]
    */
  def main(args: Array[String]): Unit = {

    val parsedCommands = parseCommandLineArgs(args)
    if (parsedCommands == null) return

    val (weights, biases) = loadNetConfigRandomlyOrFromFile(parsedCommands)

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
      savePlots(parsedCommands, trainingErrors, testErrors)
    }
  }
}
