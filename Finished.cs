using System;

public class LayerLearnData
{
    // Assuming these are the properties used in Layer class
    public double[] inputs;            // Inputs to the layer
    public double[] weightedInputs;    // Weighted sums calculated before applying activation
    public double[] activations;       // Activations after applying the activation function
    public double[] nodeValues;        // Node values used in backpropagation

    // Constructor to initialize arrays based on the size of the layer
    public LayerLearnData(int numNodesOut)
    {
        // Initialize arrays with the size of the layer's output nodes
        weightedInputs = new double[numNodesOut];
        activations = new double[numNodesOut];
        nodeValues = new double[numNodesOut];
    }
}

// IActivation Interface
public interface IActivation
{
    // Activation function applied to the weighted sum
    double Activate(double weightedSum);

    // Derivative of the activation function with respect to the weighted sum
    double Derivative(double weightedSum);
}

// ICost Interface
public interface ICost
{
    // Calculate the cost between predicted and expected outputs
    double Calculate(double predicted, double expected);

    // Derivative of the cost with respect to the predicted output
    double Derivative(double predicted, double expected);
}

//   Example ActivationSigmoid class implementing IActivation
public class ActivationSigmoid : IActivation
{
    public double Activate(double weightedSum)
    {
        return 1 / (1 + Math.Exp(-weightedSum));
    }

    public double Derivative(double weightedSum)
    {
        double sigmoid = Activate(weightedSum);
        return sigmoid * (1 - sigmoid);
    }
}

// Example QuadraticCost class implementing ICost
public class QuadraticCost : ICost
{
    public double Calculate(double predicted, double expected)
    {
        return 0.5 * Math.Pow(predicted - expected, 2);
    }

    public double Derivative(double predicted, double expected)
    {
        return predicted - expected;
    }
}

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

        //initialize the nodes coming in, with "this"
        //initialize the nodes coming out, with "this"
        //you initialize the weights as a 2d array, nodes in, by nodes out to account for each connection (as a new 2d array)
        //initialize the biases of outgoing nodes (as a new array)
        //we initialize them dynamically because we want to make sure we have the space for them


        costGradientW = new double[weights.Length];
        costGradientB = new double[biases.Length];
        weightVelocities = new double[weights.Length];
        biasVelocities = new double[biases.Length];

        InitializeRandomWeights(rng);
    }

    //calculates the activations of the output nodes based on the given input values
    //the parameter is a layer, it outputs another layer
    public double[] CalculateOutputs(double[] inputs, LayerLearnData learnData)
    {
        // initializes an array weightedInputs to store the weighted sums for each output node
        //its important to remember that the nodes coming in refer to the previous layer and nodes going out
        //refer to the number of nodes in that current layer, so in the case of a 2 3 4, the second layer would have 2 coming in and 3 going out.
        double[] weightedInputs = new double[numNodesOut]; 

        //this helps make the next layer the size of the outgoing nodes of the current layer if that makes sense



        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) //here we go through the output nodes
        {
            // calculate the weighted input for the current output node

            //set this variable to the current bias value
            double weightedInput = biases[nodeOut];

            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) //here is the inner loop where you add the weight to the equation, input(weight) + bioas
            {
                weightedInput += inputs[nodeIn] * GetWeight(nodeIn, nodeOut);
            }
            weightedInputs[nodeOut] = weightedInput; //
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

        return activations; //you're returning the actual array of the new output array
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
            costGradientW[i] = 0; //remember the gradients are usually set in the update gradients function
        }

        for (int i = 0; i < biases.Length; i++)
        {
            //update biases using gradients and momentum
            biasVelocities[i] = momentum * biasVelocities[i] - learnRate * costGradientB[i];
            biases[i] += biasVelocities[i];

            //resetting the gradient for the next iteration
            costGradientB[i] = 0; //remember the gradients are usually set in the update gradients function
        }
    }

    // calculate the node values for the final layer based on the cost derivatives
    public void CalculateOutputLayerNodeValues(LayerLearnData layerLearnData, double[] expectedOutputs, ICost cost)
    {
        double[] activations = layerLearnData.activations;
        //create an array, that has the activation values from the layerlearndata object,
        // these values (layerlearndata.activations) were initialized in the calculateoutputs function
        // these are the activations from the start

        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) //loop through the last layers nodes
        {
            //the costderivative is the objects derivative via the icost class
            double costDerivative = cost.Derivative(activations[nodeOut], expectedOutputs[nodeOut]);

            //same concept here, the original activation from the layer class, "IActivation activation", is getting it's derivative calculated
            //remember that that activation variable is like where all the activations stem from
            double activationDerivative = activation.Derivative(layerLearnData.weightedInputs[nodeOut]);

            layerLearnData.nodeValues[nodeOut] = activationDerivative * costDerivative;
            //and here is where we update the last layer
        }
    }

    // Calculate node values for hidden layers based on next layer's node values
    //remember that this function is called in the updateallgradients function, and the parameter is the next layer, 
    //as in the one after the index in the for loop
    public void CalculateHiddenLayerNodeValues(LayerLearnData layerLearnData, Layer nextLayer, double[] nextLayerNodeValues)
    {
        //parameters are the layers, as in all of them, and by layers
        // i mean one column and the nodevalues which are, the size of number of nodes out, (the output layer)

        double[] activations = layerLearnData.activations;
        //create an array, that has the activation values from the layerlearndata object,
        // these values (layerlearndata.activations) were initialized in the calculateoutputs function

        for (int nodeIndex = 0; nodeIndex < numNodesOut; nodeIndex++)
        //numnodesout = outgoing nodes in that layer
        //looping through the outgoing nodes of that layer
        //it knows the numnodesout because remember it was called in reference to a layer object
        //  nodeValues = hiddenLayer.CalculateHiddenLayerNodeValues(layers[hiddenLayerIndex + 1], nodeValues);
        {
            double weightedInputDerivativeSum = 0;
            //initializing the weightedinputderivativesum to 0, to be filled later

            // Calculate sum of weighted input derivatives from the next layer

            //For each node in the hidden layer, 
            //it calculates the sum of the derivatives of the weighted inputs from the nodes in the next layer
            for (int nextNodeIndex = 0; nextNodeIndex < nextLayer.numNodesOut; nextNodeIndex++)
            {
                double weight = nextLayer.GetWeight(nodeIndex, nextNodeIndex); //here you get the weight of the next layer
                weightedInputDerivativeSum += weight * nextLayerNodeValues[nextNodeIndex]; 
                //here you multiply the weight by the next layer node value
                //and use that to get the weightedinputderivative sum
            }

            //Within the outer loop, there is an inner loop that iterates over each output node of the next layer (nextLayer). 
            //The goal is to calculate the sum of weighted input derivatives from the next layer to the current hidden layer.

            // calculate activation derivative and node value for the hidden layer
            //first part of the equation based on the classes i wrote, that helps update the nodes
            double activationDerivative = activation.Derivative(layerLearnData.weightedInputs[nodeIndex]);
           
            layerLearnData.nodeValues[nodeIndex] = weightedInputDerivativeSum * activationDerivative;
            //here is where we update the values based on the activationderivative and the weightedinputderivativesum
        }
    }

    //update gradients 
    public void UpdateGradients(LayerLearnData layerLearnData)
    {
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
            {
                //here we update the gradients of the weights
                costGradientW[GetFlatWeightIndex(nodeIn, nodeOut)] += layerLearnData.inputs[nodeIn] * layerLearnData.nodeValues[nodeOut];
            }
            //here we update the gradients of the biases
            costGradientB[nodeOut] += layerLearnData.nodeValues[nodeOut];
        }
    }

    //gets the weight by means of a helper function
    //here you return a function of the specific node in regards to the weight array
    //after all in reference to the calculate outputs function it is looking at individual nodes
    public double GetWeight(int nodeIn, int nodeOut)
    {
        return weights[GetFlatWeightIndex(nodeIn, nodeOut)];
    }

    public int GetFlatWeightIndex(int inputNeuronIndex, int outputNeuronIndex)
    {
        // Each output neuron has 'numNodesIn' weights associated with it
        // We use the formula: weightIndex = outputNeuronIndex * numNodesIn + inputNeuronIndex
        //multiply the 2p times the # of nodes in + 1p
        return outputNeuronIndex * numNodesIn + inputNeuronIndex;

        //this function is supposed to map a 2d array to a 1d array 
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
        //this is an array of 2 layers side by side
        //layers as in measuring two consecutive ones
        //the minus one is to exclude the first layer i believe

        for (int i = 0; i < layers.Length; i++)
        {
            layers[i] = new Layer(layerSizes[i], layerSizes[i + 1], new Random());
        }
        //here we run a for loop where we run through the layer constructor and make new layers 
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
            //here we update the gradients
            //but we dont apply them until the applygradients formula
        }
        //gradient descent step: update all the weights and biases in the network
        ApplyGradients(learnRate / trainingBatch.Length);
        //reset all gradients to 0 to start over
        ClearAllGradients();
    }

    void UpdateAllGradients(DataPoint dataPoint)
    {
        //first we calculate the outputs so as to run the neural network and have it passed through and ran through
        CalculateOutputs(dataPoint.inputs);

        //this calculates the final layer of the datapoint inputs

        //create a layer array
        Layer outputLayer = layers[layers.Length - 1];

        //using that array to make an array the size of the output nodes 
        double[] nodeValues = new double[outputLayer.numNodesOut];

        
        outputLayer.CalculateOutputLayerNodeValues(dataPoint.expectedOutputs, new QuadraticCost(), nodeValues);
        //here we calculate the cost gradients for the output layer

        //telling the code to update the gradients for the final layer
        outputLayer.UpdateGradients(nodeValues);

        //this for loop is simply responsible for updating the gradients of the hidden layers
        //The loop aims to iterate over the hidden layers in reverse order, 
        //starting from the second-to-last layer (layers.Length - 2) and moving towards the first hidden layer (0).
        for (int hiddenLayerIndex = layers.Length - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--)
        {
            Layer hiddenLayer = layers[hiddenLayerIndex];
            //hidden layer represents the specific layer we're going over
            nodeValues = hiddenLayer.CalculateHiddenLayerNodeValues(layers[hiddenLayerIndex + 1], nodeValues);
            //seems the parameter in this example is the next layer
            //parameters are the layers, as in all of them, and by layers i mean one column and the nodevalues which are, 
            //the size of number of nodes out, (the output layer)
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