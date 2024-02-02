# WHAT:

## Neural-Network-Development
Currently working on the creation of AI models to use for classification and prediction purposes.



# WHY:

I'm really interested in AI models as I've used to AI to debug code before and became fascinated with the applications that AI could be used for. Upon firther research, I found that Neural Networks are the basis for a lot of different AI applications. I def wanna get into things like NLP down the road, but I'm currently working on this.

The goal with this is to use this for Celestial Body Classification and Spacecraft Trajectory Prediction.
And to learn more about constructing these models based on how well they perform each task.

# HOW: 

For starters the basic idea behind a Neural Network is to achieve a specific combination of Neurons to handle a given task. In my case, classification and prediction purposes.
Now when I "combination", I mean some neurons are preferred a bit more then others.

And we can choose our preference based off of "weights". Each neuron has a weight, and certain neurons are "heavier" and therefore more preferred. 
Each neuron also has a bias which will be explained more later but it's very useful when it comes to activation functions. 

## In the layer class 
We have nodes coming in and node going out, the nodes going in refer to the nodes coming in from the previous layer and the nodes going out refer to the number of nodes in that layer.

We declare arrays for weights, biases, cost gradients for weights, cost gradients for biases, weight velocities, & bias velocities.

Then we create an iactivation object, which is equipped with activate and derivative functions.


### Then we a have a function in the layer script that constructs the layer
It dynamically allocates the number of nodes coming in and out each layer
and makes an activationsigmoid object called activation, which has functions like activate and derivative.

The weights are then made a 2d array that are the size of the number of nodes coming in * the number of nodes going out, and the biases are made an array the size of the number of nodes going out. 

Here are some other variables that we initialize. 

costGradientW = new double[weights.Length];
costGradientB = new double[biases.Length];
weightVelocities = new double[weights.Length];
biasVelocities = new double[biases.Length];

and these variables are given there respective lengths according to the main variable (weights or biases) they were assigned, and then we intialize random weights at the end of the layer function.


### The Calculate outputs function has two parameters, that of a layer and that of a layerlearndata object
It's going to use the layer to calculate the outputs and it's simply going to update the object with new values.

First we create an array weightedInputs to store the weighted sums for each output node, loop through the outgoing nodes and create a weightedinput variable to hold the current bias value.
Then you loop through the inner nodes to access the weights, and add them to the equation
(equation: wx + b)
the bias value is gotten before the inner loop so each nodes bias value, pertains to each weight product between one nodes and rest of the nodes it's connected to.

And so the bias value corresponds to each node to node weight connection 
Since one bias value corresponds to one node and that one nodes corresponds to every other node in the next layer, in terms of weight connections.

And then there’s an activation array that’s the size of the number of nodes out
The idea there being to put each output node through the activation function 

then you simply update the object

And then return the activation array 

## In the NeuralNetwork class
We have an array of layers, that’s an array of layer objects. variable name = layers

### And then we have the constructor which has a parameter of an array that lists all the sizes of each layer

You make layers the size of each all layers minus one excluding the first.

And then you simply make a for loop going through the layers variable and it runs each layer through the layer constructor.

And therefore makes each layer, a layer object.
Equipping it with all its respective initializations and what not.

### And then we have a calculateoutputs function inside the neural network class that has a parameter of inputs, that being the first layer 

And then you create a for loop going through all the layers 
Where you create a variable that equals that layer and the outputs being calculated 

And you return that variable 

What this does is calculate the final layer, since you’re calculating the output for each layer.


### right after the CALCULATEOUTPUTS function you have a classify function that has an parameter of outputs, that benig the final layer
### and it calls a helper function, INDEXOFMAXVALUE, which simply returns the highest value within an array



### than we have a LEARN function that has parameters of a dataset of a trainingbatch, and learning rate
first we loop through each datapoint in the training batch and update the gradients, by putting each datapoint in the function UpdateAllGradients
the datapoints in context, could literally be pixels in a picture we're trying 

after that for loop, we call the APPLYGRADIENTS function with one parameter: learnrate/trainingbatch.Length

and at the end of the LEARN function we call the clear gradients function, to reset the gradients. to 9, for the next time the neuralnetwork is updated.



#### The UPDATEALLGRADIENTS function is where we update all the gradients, to be administered lately

first we calculate the outputs so as to run the neural network and have it passed through and ran through
using CalculateOutputs with dataPoint.inputs as the parameter
this gets the final layer of the inputs

than we create an array of layer objects, that is the size of the whole neural network minus the first layer
Layer outputLayer = layers[layers.Length - 1];

using that array to make an array the size of the output nodes of that objects outputs nodes

and call the calculateouputlayernodevalues function with respect to the output layer object

inside the calculate output layer node values function
where we first create an array called activations that equal to the layerlearndata activation private member value 

then you create a for loop, that runs through all the output nodes

it uses the activations array just created, in conjunction with activation and derivative class functions to update the values of the 
layerlearndata objects last layer

the next part of the updateallgradients function is call the updategradients function in relation to the outputlayer object

where the cost gradients are simply updated via a nodein nodeout nested loop, where the gradients are updated based on the 
updated layerlearndata object output layer values

then we have a for loop, where we go through the rest of the layers backwards 
within it we create a layer object representing the "next layer"

than we run a calculatehiddenlayers function that takes in the layer one away from the next layer , 
for ex: in the first loop, that function would take in the output layer and the current hidden layer, 
or the layer right next to the output layer

and within the hidden layer function 
we intialize the layerlearndata activations similar to before
