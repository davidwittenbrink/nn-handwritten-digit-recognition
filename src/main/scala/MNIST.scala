import breeze.linalg.{DenseVector, SparseVector}

/** Helper functions for loading the MNIST dataset.
  * Currently it loads MNIST data from CSV files.
  * @see [[http://pjreddie.com/projects/mnist-in-csv/]]
  * @todo Switch to binary mnist format
  * @todo Introduce validation set for model selection
  */
object MNIST {

  /** Loads an MNIST CSV file and returns the network inputs and their respective desired outputs.
    * @param file The path to CSV file
    * @return Tuple of list of network inputs as DenseVectors and list of respective outputs as SparseVectors
    */
  def loadData(file: String) = {
    val stream = getClass.getResourceAsStream(file)
    val lines = scala.io.Source.fromInputStream(stream).getLines

    val parsed = lines.foldLeft((List[DenseVector[Double]](), List[SparseVector[Double]]()))((trainingSamples, line) => {
      val (trainX, trainY) = trainingSamples
      val parts = line.split(',')
      val label = parts(0).toInt
      val input = DenseVector[Double](parts.slice(1, parts.length).map(_.toDouble))
      val targetOutput = SparseVector[Double]((0 to 9).toArray.map(x => if (x == label) 1.0 else 0.0))

      (trainX :+ input, trainY :+ targetOutput)
    })
    stream.close()
    parsed
  }

  /** Returns the training set splitted in inputs and their respective outputs
    * @return Tuple of list of network inputs as DenseVectors and list of respective outputs as SparseVectors
    */
  def getTrainingSet = loadData("/mnist_train.csv")

  /** Returns the test set splitted in inputs and their respective outputs
    * @return Tuple of list of network inputs as DenseVectors and list of respective outputs as SparseVectors
    */
  def getTestSet = loadData("/mnist_test.csv")
}
