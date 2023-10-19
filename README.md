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

And so in the Layer Class, we have a layer class that creates the layer, this where you'd specify the amount of Neurons you'd want.

We then have a CalculateOutputs function that simply calculates the outputs of a layer of inputs. So it would calculate the layer after any given layer, based off of it's weights and biases.

Then in the Neural Network Class, we have a function that makes layers by creating a for loop with the layer function in the Layer Class.
And then we have a CalculateOutputs function that runs a for loop using the previous CalculateOutputs function, to calculate the final layer.

And then there's a classify function, that lets me know which outgoing node, has the largest value. That lets me know what the network is leaning towards currently.

Then you'd create an activation function, that lets you regulate the influence of outgoing nodes.
