using System;
using System.Collections.Generic;
using System.Linq;

class Program
{
    static void Main(string[] args)
    {
        string filePath = "startype.csv";

        List<DataPoint> dataPoints = DataLoader.LoadData(filePath);
        List<DataPoint> preprocessedDataPoints = DataLoader.PreprocessData(dataPoints);

        int trainingSize = (int)(preprocessedDataPoints.Count * 0.8);
        List<DataPoint> trainingData = preprocessedDataPoints.GetRange(0, trainingSize);
        List<DataPoint> testData = preprocessedDataPoints.GetRange(trainingSize, preprocessedDataPoints.Count - trainingSize);

        // Define your network structure here. For example:
        int inputSize = 7; // Number of features in DataPoint
        int[] layerSizes = new int[] { inputSize, 10, 5, 1 }; // Example: 3 layers with 10, 5, and 1 nodes
        NeuralNetwork myNetwork = new NeuralNetwork(layerSizes);

        double learningRate = 0.01;
        int epochs = 100;

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            foreach (var dataPoint in trainingData)
            {
                // Convert DataPoint to network inputs and expected outputs
                double[] inputs = DataPointToInputs(dataPoint);
                double[] expectedOutputs = new double[] { dataPoint.StarType }; // Assuming StarType is the output

                myNetwork.Learn(new[] { inputs }, expectedOutputs, learningRate);
            }
            Console.WriteLine($"Epoch {epoch} completed.");
        }

        int correctPredictions = 0;
        foreach (var dataPoint in testData)
        {
            double[] inputs = DataPointToInputs(dataPoint);
            int predicted = myNetwork.Classify(inputs);
            if (predicted == dataPoint.StarType)
            {
                correctPredictions++;
            }
        }

        double accuracy = (double)correctPredictions / testData.Count;
        Console.WriteLine($"Accuracy: {accuracy}");
    }

    private static double[] DataPointToInputs(DataPoint dataPoint)
    {
        // Convert a DataPoint object to an array of inputs for the neural network
        return new double[]
        {
            dataPoint.AbsoluteTemperature,
            dataPoint.RelativeLuminosity,
            dataPoint.RelativeRadius,
            dataPoint.AbsoluteMagnitude,
            EncodeStarColor(dataPoint.StarColor),
            EncodeSpectralClass(dataPoint.SpectralClass)
            // Encode categorical variables (StarColor, SpectralClass) if necessary
        };
    }

    private static double EncodeStarColor(string color)
    {
        // Simple encoding for demonstration. Adjust according to your dataset.
        switch(color.ToLower())
        {
            case "white": return 1.0;
            case "red": return 2.0;
            case "blue": return 3.0;
            case "yellow": return 4.0;
            // Add other cases as needed
            default: return 0.0; // Unknown or unspecified color
        }
    }

    private static double EncodeSpectralClass(string spectralClass)
    {
        // Simple encoding for demonstration. Adjust according to your dataset.
        switch(spectralClass.ToUpper())
        {
            case "O": return 1.0;
            case "B": return 2.0;
            case "A": return 3.0;
            // Add other cases as needed
            default: return 0.0; // Unknown or unspecified class
        }
    }
}
