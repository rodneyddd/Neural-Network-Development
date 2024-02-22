using System;
using System.Collections.Generic;
using System.Linq;

class Program
{
    static void Main(string[] args)
    {
        string filePath = "startype.csv";
        //here we a filePath variable with the path to a CSV file named "startype.csv".

        List<DataPoint> dataPoints = DataLoader.LoadData(filePath); 
        //here we load the data from the CSV file into a list of DataPoint objects.
        List<DataPoint> preprocessedDataPoints = DataLoader.PreprocessData(dataPoints);
        //this part of the code preprocesses the data using a method called PreprocessData from the DataLoader class.
        
        int trainingSize = (int)(preprocessedDataPoints.Count * 0.8);
        List<DataPoint> trainingData = preprocessedDataPoints.GetRange(0, trainingSize);
        List<DataPoint> testData = preprocessedDataPoints.GetRange(trainingSize, preprocessedDataPoints.Count - trainingSize);
        //here we declare the training data and test data 

        // Define your network structure here. For example:
        int inputSize = 7; // Number of features in DataPoint
        int[] layerSizes = new int[] { inputSize, 10, 5, 1 }; // Example: 3 layers with 10, 5, and 1 nodes
        NeuralNetwork myNetwork = new NeuralNetwork(layerSizes);

        double learningRate = 0.01; //here we define a learning rate
        int epochs = 100; //here we define the iterations 


        //heres where we actually test the code
        for (int epoch = 1; epoch <= epochs; epoch++) //we loop via the amount of iterations
        {
            foreach (var dataPoint in trainingData) //go through all the datapoints in the trainingdata
            {
                // Convert DataPoint to network inputs and expected outputs
                double[] inputs = DataPointToInputs(dataPoint); //here we get the inputs based on the datapointstoinputs function
                double[] expectedOutputs = new double[] { dataPoint.StarType }; // Assuming StarType is the output

                myNetwork.Learn(new[] { inputs }, expectedOutputs, learningRate); //here we simply run the network through the learn function
            }
            Console.WriteLine($"Epoch {epoch} completed."); //here we let ourselves know when the iteration is over
        }

        //heres where we actually calculate the accuracy
        int correctPredictions = 0; //here we define a correctpredictions variable
        foreach (var dataPoint in testData) //here we go through the test data
        {
            double[] inputs = DataPointToInputs(dataPoint); //here we make an inputs array
            int predicted = myNetwork.Classify(inputs); //here we use the classify function to find out what the output was
            if (predicted == dataPoint.StarType) //and based on this if statement we increment the corrected predictions variable
            {
                correctPredictions++;
            }
        }

        double accuracy = (double)correctPredictions / testData.Count; //here we define an accuracy variable
        Console.WriteLine($"Accuracy: {accuracy}"); //and output it on the console for us to see
    }

    private static double[] DataPointToInputs(DataPoint dataPoint)
    {
        // Convert a DataPoint object to an array of inputs for the neural network
        // This creates an array with 6 indexes, each of them corresponding to a different attribute 
        //index 0 corresponds to the absolute temperature and so on
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
            case "F": return 4.0;
            case "M": return 5.0;
            case "G": return 6.0;
            case "K": return 7.0;
            // Add other cases as needed
            default: return 0.0; // Unknown or unspecified class
        }
    }
}
