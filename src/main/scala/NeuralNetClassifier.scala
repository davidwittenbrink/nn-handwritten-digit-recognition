import breeze.linalg.{DenseMatrix, DenseVector, SparseVector, Transpose, argmax}
import breeze.numerics.sigmoid
import play.api.libs.json._


/** An object containing several pure functions to solve classification tasks using neural networks
  * with Stochastic Gradient Descent as learning algorithm. The implementation sticks to the math as explained
  * in Michael Nielsen's fantastic book about neural networks.
  * For an example how you could use this classifier for optical image recognition on the MNIST data set, see
  * how it is used in the main function of this project.
  * @see
  *      [[http://neuralnetworksanddeeplearning.com]]
  *      [[Main.main]]
  *
  */
object NeuralNetClassifier {

  /** Returns a list of random weight matrices for a given list of layer sizes.
    *
    * @param layers A list of layer sizes. E.g. List(2, 3, 1) for a network with 2 input neurons, one hidden
    *               layer with 3 neurons and an output layer with 1 neuron.
    *
    * @return List of random weight matrices
    */
  def generateRandomWeightMatrices(layers: List[Int]): List[DenseMatrix[Double]] = {
    val normal01 = breeze.stats.distributions.Gaussian(0, 1)
    (1 until layers.length).toList.map(layerIdx => DenseMatrix.rand[Double](layers(layerIdx), layers(layerIdx-1), normal01))

  }

  /** Returns a list of random bias vectors for a given list of layer sizes.
    *
    * @param layers A list of layer sizes. E.g. List(2, 3, 1) for a network with 2 input neurons, one hidden
    *               layer with 3 neurons and an output layer with 1 neuron.
    *
    * @return List of random bias vectors
    */
  def generateRandomBiasVectors(layers: List[Int]): List[DenseVector[Double]] = {
    val normal01 = breeze.stats.distributions.Gaussian(0, 1)
    (1 until layers.length).toList.map(layerIdx => DenseVector.rand[Double](layers(layerIdx), normal01))
  }

  /** Returns a tuple of random weight matrices and random bias vectors for a given list of layer sizes.
    *
    * @param layers A list of layer sizes. E.g. List(2, 3, 1) for a network with 2 input neurons, one hidden
    *               layer with 3 neurons and an output layer with 1 neuron.
    *
    * @return Tuple of list of random weights and list of random biases
    */
  def generateRandomNetConfig(layers: List[Int]): (List[DenseMatrix[Double]], List[DenseVector[Double]]) = {
    (generateRandomWeightMatrices(layers), generateRandomBiasVectors(layers))
  }

  /** Returns a DenseVector with the sigmoid function applied to each element.
    *
    * @param x A DenseVector to which's elements the sigmoid function is applied.
    * @return DenseVector with the sigmoid applied element-wise
    */
  def activationFn(x: DenseVector[Double]) = sigmoid(x)

  /** Returns a DenseVector with the sigmoid derivative applied to each element.
    *
    * @param x A DenseVector to which's elements the sigmoid function derivative is applied.
    * @return DenseVector with the sigmoid derivative applied element-wise
    */
  def activationFnDerivative(x: DenseVector[Double]) = {
    val s = sigmoid(x)
    s :* (DenseVector.ones[Double](x.length) :- s)
  }

  /** Calculates the quadratic cost derivative for a given target and network output.
    *
    * Quadratic cost:
    *   (1/2) * sum( (target - networkOutput) ** 2 )
    * Quadratic cost derivative:
    *   networkOutput - target
    * @param target The desired output of the network
    * @param networkOutput The output of the network
    * @return A vector containing the costFn derivatives
    */
  def costFnDerivative(target: SparseVector[Double], networkOutput: DenseVector[Double]) = networkOutput :- target

  /** Calculates the output of a neuron for a given input vector [N X 1], the neuron weights [1 X N] and it's bias (scalar).
    *
    * @param input The input vector (dimension [N X 1])
    * @param neuronWeights The neuron weights as a row vector (dimension [1 X N])
    * @param bias The bias of the neuron
    * @return The output for the given neuron
    */
  def calcNeuronOutput(input: DenseVector[Double], neuronWeights: Transpose[DenseVector[Double]], bias: Double) = {
    neuronWeights.inner.dot(input) + bias
  }

  /** Given a config and input this function does a feedforward run and returns a tuple of
    *  - The network output
    *  - A list of z vectors calculated along the way
    *    for a detailed explanation)
    *  - A list of activation vectors (the output vectors of each layer)
    *
    * @param weights A list of the weight matrices of the network (e.g. generated by the function [[generateRandomNetConfig]])
    * @param biases A list of the bias vectors of the network (e.g. generated by the function [[generateRandomNetConfig]])
    * @param input The input to the network
    * @return Tuple of the network output, list of z vectors, list of activation vectors
    * @see [[http://neuralnetworksanddeeplearning.com/chap2.html#warm_up_a_fast_matrix-based_approach_to_computing_the_output_from_a_neural_network]]
    */
  def predict(weights: List[DenseMatrix[Double]], biases: List[DenseVector[Double]], input: DenseVector[Double]) = {
    val inputDenseVec = input.toDenseVector
    predictRecursive(weights, biases, inputDenseVec, 0, List(), List(inputDenseVec))
  }

  /** Given a config and input this function does a feedforward run and returns a tuple of
    *  - The network output
    *  - A list of z vectors calculated along the way
    *  - A list of activation vectors along the way (the output vectors of each layer)
    *
    * The output of the network is calculated in a recursive manner. To do this in an efficient (tail-recursive) way,
    * this function needs some helper arguments. If you don't want to take care of the initialisation of these arguemnts
    * yourself, the function [[predict]] has an easier signature and does this for you.
    *
    * @param weights A list of the weight matrices of the network (e.g. generated by the function [[generateRandomNetConfig]])
    * @param biases A list of the bias vectors of the network (e.g. generated by the function [[generateRandomNetConfig]])
    * @param input The input to the network
    * @param layerIdx The index of the layer in the current recursion step (0 is input layer)
    * @param zs A list of z vectors that is calculated and accumulated along the way
    * @param activations A list of activation vectors that is calculated and accumulated along the way
    * @return Tuple of the network output, list of z vectors, list of activation vectors
    * @see [[http://neuralnetworksanddeeplearning.com/chap2.html#warm_up_a_fast_matrix-based_approach_to_computing_the_output_from_a_neural_network]]
    */
  def predictRecursive(weights: List[DenseMatrix[Double]], biases: List[DenseVector[Double]],
                       input: DenseVector[Double], layerIdx: Int, zs: List[DenseVector[Double]],
                       activations: List[DenseVector[Double]]): (DenseVector[Double], List[DenseVector[Double]], List[DenseVector[Double]]) = {

    val nNextLyrNeurons = weights(layerIdx).rows

    val out = (0 until nNextLyrNeurons).map(hiddenNeuronIdx => {
      val w = weights(layerIdx)(hiddenNeuronIdx, ::)
      val b = biases(layerIdx)(hiddenNeuronIdx)
      calcNeuronOutput(input, w, b)
    })

    val z = DenseVector[Double](out.toArray)
    val activation = activationFn(z)

    if (layerIdx >= weights.length - 1) {
      // We've calculated the outputs for the output layer
      (activation, zs :+ z, activations :+ activation)
    } else {
      predictRecursive(weights, biases, activation, layerIdx + 1, zs :+ z, activations :+ activation)
    }
  }

  /** Returns a list of delta vectors needed for gradient descent.
    *
    * @param weights A list of the weight matrices of the network (e.g. generated by the function [[generateRandomNetConfig]])
    * @param biases A list of the bias vectors of the network (e.g. generated by the function [[generateRandomNetConfig]])
    * @param networkOutput The output of the network for the training sample
    * @param target The desired output of the network
    * @param zs A list of z vectors that were calculated while calculating the network output
    * @return List of delta vectors
    * @see [[http://neuralnetworksanddeeplearning.com/chap2.html]]
    */
  def calcDeltas(weights: List[DenseMatrix[Double]], biases: List[DenseVector[Double]], networkOutput: DenseVector[Double],
                 target: SparseVector[Double], zs: List[DenseVector[Double]]) = {
    recursiveCalcDeltas(weights, biases, networkOutput, target, zs, zs.length - 1, List())
  }

  /** Returns a list of delta vectors needed for gradient descent.
    *
    * The list of delta vectors is calculated in a recursive manner. To achieve this efficiently (tail-recursive),
    * some helper arguments are required. The function [[calcDeltas]] initialises these arguments correctly and offers
    * an easier signature.
    *
    * @param weights A list of the weight matrices of the network (e.g. generated by the function [[generateRandomNetConfig]])
    * @param biases A list of the bias vectors of the network (e.g. generated by the function [[generateRandomNetConfig]])
    * @param networkOutput The output of the network for the training sample
    * @param target The desired output of the network
    * @param zs A list of z vectors that were calculated while calculating the network output
    * @param layerIdx The index of the layer in the current recursion step. This should be initialised
    *                 to the length of weight matrices - 1 (meaning the nr of total layers - 2).
    * @param deltas The deltas calculated so far. This list is accumulated during the recursion steps and ultimately
    *               returned in the last step.
    * @return List of delta vectors
    * @see http://neuralnetworksanddeeplearning.com/chap2.html
    */
  def recursiveCalcDeltas(weights: List[DenseMatrix[Double]], biases: List[DenseVector[Double]], networkOutput: DenseVector[Double],
                          target: SparseVector[Double], zs: List[DenseVector[Double]], layerIdx: Int,
                          deltas: List[DenseVector[Double]]): List[DenseVector[Double]] = layerIdx match {

    case -1 => deltas
    case _ if layerIdx == (weights.length - 1) => {
      // Calculate updated weights for the output neuron
      val delta = costFnDerivative(target, networkOutput) :* activationFnDerivative(zs.last)
      recursiveCalcDeltas(weights, biases, networkOutput, target, zs, layerIdx - 1, delta +: deltas)
    }
    case _ => {
      val weightsNextLayer = weights(layerIdx + 1)
      val deltaNextLayer = deltas.head
      val z = zs(layerIdx)

      val delta = (weightsNextLayer.t * deltaNextLayer) :* activationFnDerivative(z)
      recursiveCalcDeltas(weights, biases, networkOutput, target, zs, layerIdx - 1, delta +: deltas)
    }
  }

  /** This function returns an updated tuple of weights and biases after training the network on the given
    * set of examples.
    * This function bascially implements the backpropagation algorithm.
    *
    * @param weights A list of the weight matrices of the network (e.g. generated by the function [[generateRandomNetConfig]])
    * @param biases A list of the bias vectors of the network (e.g. generated by the function [[generateRandomNetConfig]])
    * @param trainInput A list of training inputs
    * @param trainTarget A list of desired outputs corresponding to trainInput
    * @param learningRate The learning rate
    * @return Tuple of a list of trained weight matrices and a list of trained bias vectors
    * @see [[http://neuralnetworksanddeeplearning.com/chap2.html#the_backpropagation_algorithm]]
    */
  def train(weights: List[DenseMatrix[Double]], biases: List[DenseVector[Double]], trainInput: List[DenseVector[Double]],
            trainTarget: List[SparseVector[Double]], learningRate: Double) = {


    // We start by summing up the rates of changes in respect to bias/weights for each training example
    // (basically summing up the formulas BP3 and BP4 in the book over each training sample and for each layer)
    val initWeightChangeSums = weights.map(w => DenseMatrix.zeros[Double](w.rows, w.cols))
    val initBiasChangeSums = biases.map(b => DenseVector.zeros[Double](b.length))

    val changes = trainInput.indices.foldLeft((initWeightChangeSums, initBiasChangeSums))((sumsSoFar, trainDataIndex) => {
      val trainX = trainInput(trainDataIndex)
      val trainY = trainTarget(trainDataIndex)
      val (nnOutput, zs, activations) = predict(weights, biases, trainX)
      val deltas = calcDeltas(weights, biases, nnOutput, trainY, zs)

      // for given training sample, calculate delta and weight/bias changes for each layer of the network,
      // then add them to the changes we have calculated in previous examples
      weights.indices.foldLeft(sumsSoFar)((changePerLayerSums, layerIdx) => {

        val (weightSums, biasSums) = changePerLayerSums

        val delta = deltas(layerIdx)
        // The activations array is 1 element bigger than the weights array.
        // This means, that the following code gives us the activation from
        // the previous layer:
        val activation = activations(layerIdx)
        val newWeightSum = (delta * activation.t) + weightSums(layerIdx)
        val newBiasSum = delta + biasSums(layerIdx)

        (weightSums.patch(layerIdx, List(newWeightSum), 1),
          biasSums.patch(layerIdx, List(newBiasSum), 1))
      })
    })

    // After summing up the changes as mentioned, we divide the sums by the number of training samples
    // (to get the average rate of change) and multiply the learning rate. After that, we subtract the
    // calculated average changes from our current weights/biases and return the updated weights/biases.
    val (weightChanges, biasChanges) = changes
    val scaleFactor = learningRate / trainInput.length

    (weightChanges.indices.toList.map(layerIdx => weights(layerIdx) - (weightChanges(layerIdx) :* scaleFactor)),
      biasChanges.indices.toList.map(layerIdx => biases(layerIdx) - (biasChanges(layerIdx) :* scaleFactor)))
  }

  /** Returns a list of n random numbers
    *
    * @param n The number of random numbers
    * @param from The lower bound (inclusive)
    * @param to The upper bound (inclusive)
    * @return A list of n random numbers between the bounds
    */
  def randomNumbers(n: Int, from: Int, to: Int) = {
    val rnd = new scala.util.Random
    (0 until n).toList.map(_ => from + rnd.nextInt( (to - from) + 1 ))
  }

  /** Performs Stochastic Gradient Descent and returns a tuple of trained weights and biases.
    * This method has the side effect of printing the current epoch number to the terminal.
    *
    * @param weights A list of the weight matrices of the network (e.g. generated by the function [[generateRandomNetConfig]])
    * @param biases A list of the bias vectors of the network (e.g. generated by the function [[generateRandomNetConfig]])
    * @param batchSize The size of one batch
    * @param epochs The number of epochs
    * @param learningRate The learning rate
    * @param trainX A list of training inputs
    * @param trainY A list of desired outputs corresponding to the training inputs
    * @param shuffleFn A function for shuffling the input sample indices.
    *                  Can be used for tests (dependency injection)
    * @param testX A list of test sample inputs (for test error calculation)
    * @param testY A list of test sample desired outputs (for test error calculation)
    * @return Tuple of a list of trained weight matrices and a list of trained bias vectors
    */
  def sgd(weights: List[DenseMatrix[Double]], biases: List[DenseVector[Double]], batchSize: Int, epochs: Int,
          learningRate: Double, trainX: List[DenseVector[Double]], trainY: List[SparseVector[Double]],
          testX: List[DenseVector[Double]], testY: List[SparseVector[Double]],
          shuffleFn: List[Int] => List[Int] = (l) => scala.util.Random.shuffle(l)) = {

    (0 until epochs).foldLeft((weights, biases, List[Double](), List[Double]())) {
      case ((epochWeights, epochBiases, trainingErrors, testErrors), epochNr) => {
        val totalNumberOfBatches = (trainX.length / batchSize.toDouble).floor.toInt
        val shuffledIndices = shuffleFn(trainX.indices.toList)

        val (trainedWeights, trainedBiases) = (0 until totalNumberOfBatches).foldLeft((epochWeights, epochBiases)) {
          case ((w, b), batchNr) => {
            val batchIndices = shuffledIndices.slice(batchNr * batchSize, (batchNr + 1) * batchSize) //randomNumbers(10, 0, trainX.length - 1)
            val trainXBatch = batchIndices.map(trainX(_))
            val trainYBatch = batchIndices.map(trainY(_))

            train(w, b, trainXBatch, trainYBatch, learningRate)
          }
        }
        val trainingErr = calcClassificationError(trainedWeights, trainedBiases, trainX, trainY)
        val testErr = calcClassificationError(trainedWeights, trainedBiases, testX, testY)

        println(s"Epoch ${epochNr+1}/$epochs. Training error = $trainingErr Test error = $testErr")
        (trainedWeights, trainedBiases, trainingErrors :+ trainingErr, testErrors :+ testErr)
      }
    }
  }

  /** Calculates the classification error in percent (1.0 for 100%)
    *
    * @param weights A list of the weight matrices of the network (e.g. generated by the function [[generateRandomNetConfig]])
    * @param biases A list of the bias vectors of the network (e.g. generated by the function [[generateRandomNetConfig]])
    * @param testX A list of test inputs
    * @param testY A list of desired outputs corresponding to the test inputs
    * @return The percentage of classification errors
    */
  def calcClassificationError(weights: List[DenseMatrix[Double]], biases: List[DenseVector[Double]],
                              testX: List[DenseVector[Double]], testY: List[SparseVector[Double]]) = {

    val totalErrors = testX.zip(testY).foldLeft(0)((errorsSoFar, testSample) => {
      val (testInput, testTarget) = testSample
      val nnResult = predict(weights, biases, testInput)._1
      val detectedClass = argmax(nnResult)
      val correctClass = argmax(testTarget)
      if (detectedClass != correctClass) errorsSoFar + 1 else errorsSoFar
    })
    totalErrors / testX.length.toDouble
  }

  /** Returns a JSON string containing the network configuration. The JSON is in the same format as the snapshots in
    * Carmen Popoviciu's JavaScript neuralnet implementation to allow importing/exporting networks.
    *
    * @param weights A list of the weight matrices of the network (e.g. generated by the function [[generateRandomNetConfig]])
    * @param biases A list of the bias vectors of the network (e.g. generated by the function [[generateRandomNetConfig]])
    * @return A JSON string containing the network configuration
    * @see [[https://github.com/CarmenPopoviciu/neural-net]]
    */
  def getJsonSnapshot(weights: List[DenseMatrix[Double]], biases: List[DenseVector[Double]]) = {

    val nInputNeurons = weights.head.cols
    val firstLayerJson = Json.obj(
      "nodes" -> (0 until nInputNeurons).map(_ => Json.obj("weights" -> JsNull, "bias" -> JsNull))
    )

    val layersJson = weights.indices.map(layerIdx => {
      val nLayerNeurons = weights(layerIdx).rows
      val nodes = (0 until nLayerNeurons).map(neuronIdx => {
        val layerWeightsVec = weights(layerIdx)(neuronIdx, ::).inner
        val bias = biases(layerIdx)(neuronIdx)

        Json.obj("bias" -> bias, "weights" -> layerWeightsVec.toArray)
      })
      Json.obj("nodes" -> nodes)
    })

    Json.obj(
      "layers" -> (firstLayerJson +: layersJson)
    ).toString()
  }

  /** Returns a tuple of a list of weight matrices and a list of bias vectors from a JSON string. The JSON has to be in the
    * same format as the snapshots of Carmen Popoviciu's JavaScript neuralnet implementation.
    *
    * @param jsonString A JSON string containing the weights and biases of the network in the format of
    *                   Carmen Popoviciu's JavaScript neuralnet implementation.
    * @return Tuple of a list of parsed weight matrices and a list of parsed bias vectors
    * @see [[https://github.com/CarmenPopoviciu/neural-net]]
    */
  def getConfigFromSnapshot(jsonString: String) = {
    val jsonObj = Json.parse(jsonString)
    val layers = (jsonObj \ "layers").as[Seq[JsValue]]

    (1 until layers.length).foldLeft((List[DenseMatrix[Double]](), List[DenseVector[Double]]()))((weightsAndBiases, layerIdx) => {
      val (ws, bs) = weightsAndBiases
      val nodes = (layers(layerIdx) \ "nodes").as[Seq[JsValue]]
      val nodesPrevLayer = (layers(layerIdx - 1) \ "nodes").as[Seq[JsValue]]
      val nodeWeights = nodes.map(node => DenseVector[Double]((node \ "weights").as[Array[Double]]).asDenseMatrix)
      val nodeBiases = DenseVector[Double](nodes.map(node => (node \ "bias").as[Double]).toArray)
      val weightMatrix = DenseMatrix.vertcat[Double](nodeWeights:_*)

      (ws :+ weightMatrix, bs :+ nodeBiases)
    })
  }
}