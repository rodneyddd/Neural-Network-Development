using System;

public class Layer
{

    //initializing variables
    public int numNodesIn;
    public int numNodesOut;
    public double[] weights;
    public double[] biases;
    public double[] costGradientW;
    public double[] costGradientB;
    public double[] weightVelocities;
    public double[] biasVelocities;
    public IActivation activation;

    //constructing a neural network layer
    public Layer(int numNodesIn, int numNodesOut, System.Random rng)
    {
        this.numNodesIn = numNodesIn;
        this.numNodesOut = numNodesOut;
        activation = new ActivationSigmoid();

        //creating a 2d array of all weights
        weights = new double[numNodesIn * numNodesOut];
        //creating an array of biases
        biases = new double[numNodesOut];

        costGradientW = new double[weights.Length];
        costGradientB = new double[biases.Length];
        weightVelocities = new double[weights.Length];
        biasVelocities = new double[biases.Length];

        InitializeRandomWeights(rng);
    }

    //calculates the activations of the output nodes based on the given input values and the weights and biases of the layer.

    public double[] CalculateOutputs(double[] inputs, LayerLearnData learnData)
    {
        // an array of weightedinputs
        double[] weightedInputs = new double[numNodesOut];

        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            // calculate the weighted input for the current output node
            double weightedInput = biases[nodeOut];

            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
            {
                weightedInput += inputs[nodeIn] * GetWeight(nodeIn, nodeOut);
            }
            weightedInputs[nodeOut] = weightedInput;
        }

        //putting it through the activation function
        double[] activations = new double[numNodesOut];
        for (int outputNode = 0; outputNode < numNodesOut; outputNode++)
        {
            activations[outputNode] = activation.Activate(weightedInputs, outputNode);
        }

        // Store the values in the provided learnData object
        learnData.inputs = inputs;
        learnData.weightedInputs = weightedInputs;
        learnData.activations = activations;

        return activations;
    }


    //applying gradients to update weights and biases during training
    public void ApplyGradients(double learnRate, double regularization, double momentum)
    {
        double regularizationFactor = 1.0 - learnRate * regularization;

        //updating the weights and biases during training
        for (int i = 0; i < weights.Length; i++)
        {
            //update weights using gradients and momentum
            weightVelocities[i] = momentum * weightVelocities[i] - learnRate * costGradientW[i];
            weights[i] = regularizationFactor * weights[i] + weightVelocities[i];

            //resetting the gradient for the next iteration
            costGradientW[i] = 0;
        }

        for (int i = 0; i < biases.Length; i++)
        {
            //update biases using gradients and momentum
            biasVelocities[i] = momentum * biasVelocities[i] - learnRate * costGradientB[i];
            biases[i] += biasVelocities[i];

            //resetting the gradient for the next iteration
            costGradientB[i] = 0;
        }
    }

    // calculate the node values for the final layer based on the cost derivatives
    public void CalculateOutputLayerNodeValues(LayerLearnData layerLearnData, double[] expectedOutputs, ICost cost)
    {
        double[] activations = layerLearnData.activations;

        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            double costDerivative = cost.Derivative(activations[nodeOut], expectedOutputs[nodeOut]);
            double activationDerivative = activation.Derivative(layerLearnData.weightedInputs[nodeOut]);

            layerLearnData.nodeValues[nodeOut] = activationDerivative * costDerivative;
        }
    }

    // Calculate node values for hidden layers based on next layer's node values
    public void CalculateHiddenLayerNodeValues(LayerLearnData layerLearnData, Layer nextLayer, double[] nextLayerNodeValues)
    {
        double[] activations = layerLearnData.activations;

        for (int nodeIndex = 0; nodeIndex < numNodesOut; nodeIndex++)
        {
            double weightedInputDerivativeSum = 0;

            // Calculate sum of weighted input derivatives from the next layer
            for (int nextNodeIndex = 0; nextNodeIndex < nextLayer.numNodesOut; nextNodeIndex++)
            {
                double weight = nextLayer.GetWeight(nodeIndex, nextNodeIndex);
                weightedInputDerivativeSum += weight * nextLayerNodeValues[nextNodeIndex];
            }

            // calculate activation derivative and node value for the hidden layer
            double activationDerivative = activation.Derivative(layerLearnData.weightedInputs[nodeIndex]);
            layerLearnData.nodeValues[nodeIndex] = weightedInputDerivativeSum * activationDerivative;
        }
    }

    //
    public void UpdateGradients(LayerLearnData layerLearnData)
    {
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
            {
                costGradientW[GetFlatWeightIndex(nodeIn, nodeOut)] += layerLearnData.inputs[nodeIn] * layerLearnData.nodeValues[nodeOut];
            }

            costGradientB[nodeOut] += layerLearnData.nodeValues[nodeOut];
        }
    }
    //gets the weight by means of a helper function
    public double GetWeight(int nodeIn, int nodeOut)
    {
        return weights[GetFlatWeightIndex(nodeIn, nodeOut)];
    }

    public int GetFlatWeightIndex(int inputNeuronIndex, int outputNeuronIndex)
    {
        // Each output neuron has 'numNodesIn' weights associated with it
        // We use the formula: weightIndex = outputNeuronIndex * numNodesIn + inputNeuronIndex
        return outputNeuronIndex * numNodesIn + inputNeuronIndex;
    }

    //chooses the best activation function for the best situation
    public void SetActivationFunction(IActivation activation)
    {
        this.activation = activation;
    }

    //simplify gets a random weight
    public void InitializeRandomWeights(System.Random rng)
    {
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = rng.NextDouble() * 2 - 1;
        }
    }
}


public class NeuralNetwork
{
    Layer[] layers;

    //responsible for the connection of multiple layers ultimately creating the Neural Network B)
    public NeuralNetwork(params int[] layerSizes)
    {
        layers = new Layer[layerSizes.Length - 1];

        for (int i = 0; i < layers.Length; i++)
        {
            layers[i] = new Layer(layerSizes[i], layerSizes[i + 1], new Random());
        }
    }

    //calculates the absolute final output layer, by looping through each layer with the calculate outputs function
    public double[] CalculateOutputs(double[] inputs)
    {
        foreach (Layer layer in layers)
        {
            inputs = layer.CalculateOutputs(inputs);
        }
        return inputs;
    }

    //this function is created so that we can return the node that the Neural Network has the highest confidence in
    //this is to help gauge at any given point how well trained the Neural Network is
    public int Classify(double[] inputs)
    {
        double[] outputs = CalculateOutputs(inputs);
        //creating an array of outputs within an array, that we find the highest value of within 
        return IndexOfMaxValue(outputs);
    }

    //this is the implemented learn functon
    public void Learn(DataPoint[] trainingBatch, double learnRate)
    {
        //takes the current training batch
        //adds up the gradients for each of them
        foreach (DataPoint dataPoint in trainingBatch)
        {
            UpdateAllGradients(dataPoint);
        }
        //gradient descent step: update all the weights and biases in the network
        ApplyGradients(learnRate / trainingBatch.Length);
         //reset all gradients to 0 to start over
        ClearAllGradients();
    }

    void UpdateAllGradients(DataPoint dataPoint)
    {
        //each layer will store the values we need, such as the weighted inputs and activations
        //this is so that the data will be saved in the layer script
        CalculateOutputs(dataPoint.inputs);

        //create a layer array
        Layer outputLayer = layers[layers.Length - 1];
        //using that array to store the parts 
        double[] nodeValues = new double[outputLayer.numNodesOut];

        
        outputLayer.CalculateOutputLayerNodeValues(dataPoint.expectedOutputs, new QuadraticCost(), nodeValues);

        //telling the code to update the gradients for all the weights and biases
        outputLayer.UpdateGradients(nodeValues);

        //this for loop is simply responsible for updating the gradients of the hidden layers
        for (int hiddenLayerIndex = layers.Length - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--)
        {
            Layer hiddenLayer = layers[hiddenLayerIndex];
            nodeValues = hiddenLayer.CalculateHiddenLayerNodeValues(layers[hiddenLayerIndex + 1], nodeValues);
            hiddenLayer.UpdateGradients(nodeValues);
        }
    }
    //this function as named applies the gradients and the learn rate controls the step size of the gradient descent
    void ApplyGradients(double learnRate)
    {
        //just applies the gradients
        foreach (Layer layer in layers)
        {
            //to be given legitimate values upon object creation
            layer.ApplyGradients(learnRate, 0.0, 0.0);
        }
    }

    //this function simply clears the gradients
    void ClearAllGradients()
    {
        foreach (Layer layer in layers)
        {
            Array.Clear(layer.costGradientW, 0, layer.costGradientW.Length);
            Array.Clear(layer.costGradientB, 0, layer.costGradientB.Length);
        }
    }

    //this function is responsible for helping calculate which node is the most influential
    //this is where you'd plug in a node layer, usually the last one and it tells you which node has the highest influence
    int IndexOfMaxValue(double[] array)
    {
        int maxIndex = 0;
        double maxValue = array[0];

        for (int i = 1; i < array.Length; i++)
        {
            if (array[i] > maxValue)
            {
                maxIndex = i;
                maxValue = array[i];
            }
        }
        return maxIndex;
    }
}

/*

Layer Class:

Represents a single layer of a neural network.
It has variables for the number of input and output nodes, weight and bias arrays, cost gradients, weight velocities, and an activation function.
The constructor initializes the layer with the specified number of input and output nodes, and it initializes the weights randomly.
CalculateOutputs method computes the activations of the output nodes based on input values, weights, and biases, and applies the activation function.
ApplyGradients method updates weights and biases during training using the gradient descent algorithm.
CalculateOutputLayerNodeValues computes node values for the output layer based on cost derivatives.
CalculateHiddenLayerNodeValues computes node values for hidden layers based on the next layer's node values.
UpdateGradients method updates gradients during training.
There are other helper methods for weight management and activation function.

NeuralNetwork Class:

Represents the entire neural network, consisting of multiple layers.
The constructor initializes the network with the specified layer sizes and creates individual layers.
CalculateOutputs method computes the final output layer by looping through each layer.
Classify method returns the index of the output node with the highest confidence, useful for classification tasks.
Learn method updates the gradients for a batch of training data, applies gradient descent, and clears gradients.
Various helper methods like UpdateAllGradients, ApplyGradients, ClearAllGradients, and IndexOfMaxValue.
Customizable  aspects of training, including learning rate, regularization, and momentum.
To use this neural network framework, you would create an instance of the NeuralNetwork class, provide training data, and continue calling the Learn method to train the network. The trained network can then be used for making predictions or classifications.

The goal is to to use this for Spacecraft Trajectory Prediction and Celestial Body Classification

But after a couple months of tutorials, research and good old fashioned hard work here we are! My first Neural Network. 
Now to test it.
>: )

*/