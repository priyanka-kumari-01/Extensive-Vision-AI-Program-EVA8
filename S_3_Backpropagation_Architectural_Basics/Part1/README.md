# Session 3 - Backpropagation and Architectural Basics

![Task](https://raw.githubusercontent.com/pandian-raja/EVA8/main/Resources/S3/s3_part_1.png)

[Excel Sheet](NNBackpropgation.xlsx)
![Screenshot](https://raw.githubusercontent.com/pandian-raja/EVA8/main/Resources/S3/backpropgation_output.png)


---

## Part 1: Neural Networks Backpropagation

The structure and function of the human brain inspire a machine learning model called neural networks. It comprises layers of interconnected nodes or "neurons" that can learn and make predictions or decisions based on input data. The most commonly used layers in neural network architectures include the following:
	1. Input layers: It receives the input data and passes it to the network's next layer(s).
	2. Hidden layers: It performs computations on the data.
	3. Output layers: It produces the final output of the network, which can be a prediction, a class label, or some other value.
Other essential components are:
	1. Weights: During the training process, the parameters are learned.
	2. Error: This calculation in a neural network refers to determining how well the network's predictions match the actual labels of the input data.

In the below diagram, two inputs: i1 and i2, have two hidden layers: h1 and h2, and two output Layers o1 and o2. It also have weights w1, w2, w3, w4, w5, w6, w7 and w8 and Error calculated as E_Total.

## Step 1: 
	h1 = w1*i1 + w2*¡2
The output(h1) of the neuron is calculated by taking the dot product of the input vector (i1, i2) and the weight vector (w1, w2). The dot product is a mathematical operation that multiplies the first element of the input vector(i1) with the first element of the weight vector(w1), then multiplies the second element of the input vector(i2) with the second element of the weight vector(w2) and adds the results together.

	h2 = w3*i1 + w4*¡2
The output(h2) of the neuron is calculated by taking the dot product of the input vector (i1, i2) and the weight vector (w3, w4), the same as the h1 output explained.
	
	a_h1 = 𝝈(h1) = 1/(1 + exp(-h1))
The equation represents the application of an activation function 𝝈 to the output h1 of a neuron in the first hidden layer of a neural network. The activation function 𝝈 is the sigmoid function. The sigmoid function is a smooth and continuously differentiable function that maps any real-valued number to a value between 0 and 1. It has an "S" shaped curve and is defined as: 𝝈(x) = 1 / (1 + exp(-x)),  where x is the input to the function.

	a_h2 = 𝝈(h2)= 1/(1 + exp(-h2))
The equation represents the application of an activation function 𝝈 to the output h2 of a neuron in the second hidden neural network layer, the same as the a_h1 explained.
	
	o1 = w5*a_h1 + w6*a_h2
The output(o1) is the addition of the weights "w5" and "w6," which are scalar values. These are multiplied by the input values "a_h1" and "a_h2" respectively, representing the activations from the previous layer of neurons.

	o2 = w7*a_h1 + w8*a_h2
In this scalar value changes to (w7, w8) and then is multiplied with input values "a_h1" and "a_h2," respectively. The addition of these weights gives the output (o2).  

	E_total = E1 + E2
The variable "E_total" represents the total error of the network, which is the sum of two individual errors, "E1" and "E2".

	E1=½* (t1-a_01)^2 and E2 = ½ * (t2 -a_01)^2
"E1" and "E2" are expressions for the individual errors of the first and second output neurons, respectively. The error is calculated using the mean squared error (MSE) function, where "t1" and "t2" are the target values for the first and second output neurons, and "a_01" and "a_02" are the actual output values as mentioned above.

The equations for E1 and E2 are half the difference between the target value and the actual output squared. These loss functions are used to measure the network performance and update the network weights to minimize the total error. This process is called backpropagation and is a fundamental aspect of training neural networks.	

## Step 2:

	∂E_total/∂w5 = ∂(E1+E2)/∂w5 = ∂E1/∂w5(E2 path is not used)
The network's total(gradient) error with respect to the weight "w5". As shown earlier, "E_total = E1+E2", so the (E1+E2) error with respect to the weight "w5". In this case, there is no path between E2 and the weight w5, so removing the E2 error from E_total. The final equation is an E1 error with respect to the weight "w5".   

	∂E_total/∂w5 = = ∂E1/∂w5 = ∂E1/∂a_o1*∂a_o1/∂o1*∂o1/∂w5
The equation of ∂E_total/∂w5 implies the product of the change in the error of the first output with respect to the activation of the first output, the change in the activation of the first output with respect to the output, and the change in the output with respect to weight w5. 
		
	∂E1/∂a_o1 = ∂(1/2*(t1-a_o1)^2)/∂a_o1 = (a_o1 - t1)
The equation ∂E1/∂a_o1 is calculated by taking the derivative of the error function (1/2*(t1-a_o1)^2) with respect to a_o1, where t1 is the target value for the first output as a result is (a_o1-t1) which represent how much the error is affected by the change in the activation of the first output node.

	∂a_o1/∂o1 = ∂(𝝈(o1))/∂o1 = a_o1*(1-a_o1)
The equation is the derivative of the sigmoid activation function with respect to its input, o1. The derivative of the sigmoid function with respect to x is represented as 𝝈'(x) = 𝝈(x) * (1 - 𝝈(x)), where a_o1 = 𝝈(o1). This is also represented as the above expression ∂a_o1/∂o1 = ∂(𝝈(o1))/∂o1 = a_o1*(1-a_o1)

	∂o1/∂w5 = a_h1
The equation is the derivative of the output of a neuron, o1, with respect to weight, w5. In this case, the derivative is equal to the activation of the neuron, a_h1, in the previous layer. 

## Step 3:
	∂E_total/∂w5 = (a_o1 - t1)*a_o1*(1-a_o1)*a_h1
The equation represents the partial derivative of the total error (E_total) with respect to the weight (w5) of the connection between the first hidden layer (a_h1) and the first output layer (a_o1). The error term (a_o1 - t1) represents the difference between the predicted output (a_o1) and the target output (t1), and  a_o1*(1-a_o1) denotes the derivative of the sigmoid activation function applied to the first output layer.

	∂E_total/∂w6 = (a_o1 - t1)*a_o1*(1-a_o1)*a_h2
	∂E_total/∂w7 = (a_o2 - t2)*a_o2*(1-a_o2)*a_h1
	∂E_total/∂w8 = (a_o2 - t2)*a_o2*(1-a_o2)*a_h2
∂E_total/∂w6, ∂E_total/∂w7, and ∂E_total/∂w8 are also the partial derivative of the total error (E_total) with respect to the weight (w5) of the connection between the hidden layer and output layer. The error term represents the difference between the predicted output and the target output and the derivative of the sigmoid activation function applied to the output layer.

## Step 4:
	∂E1/∂a_h1 = (a_o1 - t1)*a_o1*(1-a_o1)*w5
This equation represents the partial derivative of the total error (E_total) with respect to the output of the first hidden layer (a_h1). The first term on the right-hand side, (a_o1 - t1)*a_o1*(1-a_o1)*w5, represents the sensitivity of the error of the first output layer (E1) with respect to the first hidden layer output. 

	∂E2/∂a_h2= (a_o2 - t2)*a_o2*(1-a_o2)*w7
This equation represents the partial derivative of the total error (E_total) with respect to the output of the second hidden layer (a_h2). The first term on the right-hand side represents the second output layer (E2) with respect to the first hidden layer output. 

	∂E_total/∂a_h1 = (a_o1 - t1)*a_o1*(1-a_o1)*w5 + (a_o2 - t2)*a_o2*(1-a_o2)*w7
The error of the second output layer (E1) with respect to the first hidden layer output and the error of the first output layer (E1) with respect to the second hidden layer output. The sum of these two terms gives the total error with respect to the output of the first hidden layer.

	∂E_total/∂a_h2 = (a_o1 - t1)*a_o1*(1-a_o1)*w6 + (a_o2 - t2)*a_o2*(1-a_o2)*w8
The error of the second output layer (E2) with respect to the first hidden layer output and the second hidden layer output. The sum of these two terms gives the total error with respect to the output of the first hidden layer.

## Step 5:
	∂E_total/∂w1 = ∂E_total/∂a_h1*∂a_h1/∂h1*∂h1/∂w1
This equation represents the partial derivative of the total error (E_total) with respect to the weight (w1) of the connection between the input layer (i1) and the first hidden layer (a_h1). The term ∂E_total/∂a_h1 represents the total error with respect to the output of the first hidden layer, ∂a_h1/∂h1 represents the derivative of the activation function applied to the first hidden layer output with respect to the input of the activation function, and ∂h1/∂w1 represents the sensitivity of the first hidden layer input with respect to the weight w1. 

	∂E_total/∂w2 = ∂E_total/∂a_h1*∂a_h1/∂h1*∂h1/∂w2
	∂E_total/∂w3 = ∂E_total/∂a_h2*∂a_h2/∂h2*∂h2/∂w3
	∂E_total/∂w4= ∂E_total/∂a_h2*∂a_h2/∂h2*∂h2/∂w4

∂E_total/∂w2, ∂E_total/∂w3, ∂E_total/∂w4 imples same as the ∂E_total/∂w1 with changes in variables.


## Step 6:
	∂E_total/∂w1= ((a_o1 - t1)*a_o1*(1-a_o1)*w5 + (a_o2 - t2)*a_o2*(1-a_o2)*w7)*a_h1*(1-a_h1)*i1
This equation represents the partial derivative of the total error (E_total) with respect to the weight (w1) of the connection between the input layer (i1) and the first hidden layer (a_h1). 

The first term in the parenthesis, (a_o1 - t1)a_o1(1-a_o1)*w5, represents the error of the first output layer with respect to the first hidden layer output. The second term, (a_o2 - t2)a_o2(1-a_o2)w7, represents the sensitivity of the error of the second output layer with respect to the first hidden layer output. 

The product of these two terms and the a_h1(1-a_h1) term, which is the derivative of the sigmoid activation function applied to the first hidden layer, gives the total error with respect to the output of the first hidden layer and the last term i1 represents the sensitivity of the first hidden layer input with respect to the weight w1. 

	∂E_total/∂w2 =  ((a_o1 - t1)*a_o1*(1-a_o1)*w5 + (a_o2 - t2)*a_o2*(1-a_o2)*w7)*a_h1*(1-a_h1)*i2
	∂E_total/∂w3= ((a_o1 - t1)*a_o1*(1-a_o1)*w6 + (a_o2 - t2)*a_o2*(1-a_o2)*w8)*a_h2*(1-a_h2)*i1
	∂E_total/∂w4 =  ((a_o1 - t1)*a_o1*(1-a_o1)*w6 + (a_o2 - t2)*a_o2*(1-a_o2)*w8)*a_h2*(1-a_h1)*i2
	
*∂E_total/∂w2, ∂E_total/∂w3, ∂E_total/∂w4 imples same as the ∂E_total/∂w1  with changes in variables.*