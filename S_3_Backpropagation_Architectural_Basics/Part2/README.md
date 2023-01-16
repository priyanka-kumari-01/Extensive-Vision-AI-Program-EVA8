## Part2:


![Task](https://raw.githubusercontent.com/pandian-raja/EVA8/main/Resources/S3/s3_part_2.png)


---

### 1. How many layers: Neural networks have three layers

1. Input layers: It receives the input data and passes it to the network's next layer(s).

2. Hidden layers: It performs computations on the data.

3.  Output layers: It produces the final output of the network, which can be a prediction, a class label, or some other value.

---

### 2. MaxPooling: 

Max pooling extracts the maximum value in a given set of numbers. For example, there is input with a 4x4 matrix which means 16 total input numbers, and a max pooling of 2x2 filter applied on top of it; as a result, it would be 2x2, which means 4 total output numbers. In this case, the 2x2 filter goes to the top of the first part of the 4x4 input matrix and extracts a single value, and this process continues until it reaches the last part of the 4x4 matrix of input.     

The best practice is to use max pooling starting layers of the neural network because it exacts the only maximum value in a given set of numbers and discards the remaining value. The disadvantage of using max pooling in the last layer is training a model with features that will be used to detect the object. These features are essential for neurons because when we detect dogs, we need both features, one that tells it's a dog and the other that means it's not a dog. Using max pooling at the last layer is like wasting resources on the training model as, in the end, we are just throwing away all the features. 

---

### 3 & 4. 1x1 Convolutions vs 3x3 Convolutions: 

#### 1x1 Convolutions:
 It multiplies with the whole channel instead of convolving on every input and producing a new output by mixing the features. 
 1. It combines or merges the pre-existing features of the channels found together and separates the different class features. For example, car class features like the wheel, doors, and windows are not mixed with dogs class features like tail, whisker, and four legs.  
 2. RF: I believe; when we applied the 1x1 filter on the input image, we increased RF by 1. 
 3. Less expansive and less computational method because have one parameter.
 4. Dimension reduction: we are multiplying each channel with the 1x1 filter and reducing the channel.
 5. 1x1 allows communication between all the channels with each other.  
	
#### 3x3 Convolutions: 
It has a 3x3 filter, goes on top of the 3x3 part of the input, and implies matrix multiplication. Using this 3x3 filter, we can apply it to the vertical and horizontal filters according to the requirement.
1. It doesn't combine the channel but extracts and filters the channels' features.
2. RF: When we applied the 3x3 filter on the input image, we increased RF by 2. 
3. Expansive and computational methods because they have nine parameters.

*Common: Both can be used according to the number of channels. For example, if we have 3 channel input image, we can use 1x1x3 and 3x3x3.*

---

### 5. Receptive Field: 

Receptive Field:The receptive field of a neuron in a convolutional neural network (CNN) is the region of the input image or feature map that the neuron is to "see." In the case of CNN, the receptive field of a neuron is determined by combining the convolution operations and pooling operations applied to the input image as it is propagated through the network.

The receptive field of a neuron can be controlled by the kernel size, stride, and padding of the convolutional layers, as well as the pooling size and stride of the pooling layers. As the input image or feature map is propagated through the network, the receptive field of a neuron generally increases, allowing the neuron to "see" more of the input image or feature map.

It's important to note that the receptive field size is different than the output size of an activation map. The receptive field size gives the context of the neuron, the area of the input image that affects the output. In contrast, the output size is the resulting image after applying the convolutional operation.

---

### 6. SoftMax: 
Softmax is a function that is commonly used in the output layer of a neural network to convert the output of the network into a probability distribution. The output of a neuron in the output layer is often a real number, but in order to interpret the output as a probability, the output needs to be transformed into a value between 0 and 1. The softmax function does this by applying the exponential function to each output and then normalizing the results so that they add up to 1.

The equation for the softmax function is as follows: softmax(x_i) = exp(x_i) / Σ exp(x_j)

where x_i is the output of the ith neuron and Σ exp(x_j) is the sum of the exponential of all outputs in the output layer.

The softmax function is often used in classification problems with multiple classes, where the goal is to assign an input to one of the classes. The output of the softmax function is a probability distribution over the different classes, and the class with the highest probability is chosen as the final prediction.

---

### 7. Learning Rate:
The learning rate is a hyperparameter in machine learning that controls the step size at which the optimizer makes updates to the model's parameters. In other words, it determines how fast or slow a model learns from the training data.

A high learning rate means that the optimizer will make large updates to the model's parameters with each iteration, which can lead to fast convergence but also the risk of overshooting the optimal solution or even diverging. A low learning rate means that the optimizer will make small updates to the model's parameters, which can lead to slow convergence but also a higher chance of finding the optimal solution.

The learning rate can be set manually or adaptively, but an appropriate learning rate is crucial for a neural network to learn effectively. If the learning rate is too high, the model may not converge or may converge to a sub-optimal solution, while if the learning rate is too low, the model may converge too slowly or may get stuck in a sub-optimal solution.

There are some techniques like learning rate schedules or Adaptive Learning Rate methods which adjust the learning rate during training to improve the model's performance. These methods usually starts with a high learning rate and decrease it over time, allowing the model to converge quickly at the beginning and then fine-tune the parameters as the training progresses.

---

### 8. Kernels and how do we decide the number of kernels? 
In a convolutional neural network (CNN), a kernel (also known as a filter) is a small, fixed-size matrix of weights that is used to extract features from the input image or feature map. The kernel is slid over the input, element-wise multiplied with the region of the input that it is currently "looking at", and then the results are summed up to create the output of the convolution operation. Each kernel is designed to detect a specific type of feature in the input, such as edges, textures, or patterns.

The number of kernels in a convolutional layer is a hyperparameter that can be chosen by the user. A larger number of kernels can result in the extraction of more complex and diverse features from the input, but it also increases the number of parameters in the model and the computational complexity. In practice, the number of kernels is often determined by the complexity of the task, the size of the input, and the computational resources available.

A general rule of thumb is to start with a small number of kernels, such as 32 or 64, and then increase it if necessary. The number of kernels can also be increased as the input image is propagated through the network, as the features become more abstract and complex.

It's also important to note that the number of kernels in a layer should be the same as the number of channels in the input, because each kernel is applied to each channel of the input

---

### 9. Batch Normalization: 
Batch normalization is a technique used in deep neural networks to normalize the activations of a layer, in order to stabilize and speed up the training process. It is typically applied to the outputs of a fully connected layer or a convolutional layer, before the activation function is applied.

The basic idea behind batch normalization is to normalize the activations of a layer by subtracting the mean and dividing by the standard deviation, computed over the current mini-batch. This normalization is done independently for each feature map, so that the output of the normalization has zero mean and unit variance. After normalization, the network learns a set of two parameters per activation: a scale parameter and a shift parameter, which are applied to the normalized activations. These parameters are learned during training and the normalization is typically not applied during inference.

Batch normalization has several benefits:
1. It helps to reduce the internal covariate shift, which is the change in the distribution of the inputs to a layer caused by the change in the parameters during training.
2. It helps to improve the stability of the training process by reducing the dependence of the output on the initial values of the parameters.
3. It helps to accelerate the training process by making the optimization more stable and robust, which allows the use of larger learning rates.

---

### 10. Image Normalization:
Image normalization is a technique used to adjust the brightness and contrast of an image, so that it has zero mean and unit variance. This is done by subtracting the mean of the image pixels and dividing by the standard deviation of the image pixels. The result is a normalized image that has pixel values between -1 and 1.

---

### 11. Position of MaxPooling:

In a convolutional neural network (CNN), the position of the max pooling layer can have an impact on the performance of the network. Typically, max pooling layers are placed immediately after a convolutional layer, in order to down-sample the feature maps and reduce the dimensionality of the data.

In general, max pooling layers are placed immediately after the convolutional layers to reduce the spatial dimensionality of the feature maps, in order to reduce the computational cost of the network and also make the features more robust to small translations of the input.

It is important to note that max pooling can also be used after multiple consecutive convolutional layers, this is called deep max pooling, this can help to extract more abstract and complex features from the input.

Additionally, the position of max pooling can also be used to control the size of the receptive field. As you know, the receptive field of a neuron is the area of the input image that contributes to the computation of the neuron's output. As the input image is propagated through the network, the receptive field of a neuron generally increases, allowing the neuron to "see" more of the input image.

---

### 12. Concept of Transition Layers:

In a convolutional neural network (CNN), a transition layer is a type of layer that is used to reduce the number of channels and the spatial resolution of the feature maps. This can be used to control the number of parameters in the model, as well as the computational complexity.

A common type of transition layer is a combination of a 1x1 convolution layer followed by a 2x2 max pooling layer with stride 2, this is called a transition down block. This combination of layers is used to reduce the number of channels in the feature maps and also reduce the spatial resolution by half.

The main purpose of using transition layers is to make the model computationally more efficient, while still preserving the important features learned by the network. Transition layers are typically used after several consecutive convolutional layers, in order to reduce the number of parameters and computational cost of the network.

Another use of transition layers is to make the network more robust to small translations of the input, as the receptive field of the neurons increases as the input image is propagated through the network.

---

### 13. Position of Transition Layer:
In a convolutional neural network (CNN), the position of the transition layer can have an impact on the performance of the network. Transition layers are typically used after several consecutive convolutional layers, in order to reduce the number of parameters and computational cost of the network, while still preserving the important features learned by the network.

In general, transition layers are placed after a few layers of convolutional layers, this is called "dense block" and its purpose is to extract more abstract and complex features from the input.

The dense block is followed by a transition layer, which reduce the number of channels and the spatial resolution of the feature maps. This can be used to control the number of parameters in the model, as well as the computational complexity.

Additionally, the position of transition layers can also be used to control the size of the receptive field. As the input image is propagated through the network, the receptive field of a neuron generally increases, allowing the neuron to "see" more of the input image.

---

### 14. DropOut:
Dropout is a regularization technique used in deep neural networks to prevent overfitting by randomly dropping out (i.e., setting to zero) a certain percentage of neurons during training. This means that for each training iteration, some neurons will not be updated, and their output will not be used for the computation of the loss function. This helps to reduce the co-adaptation of the neurons, which is a phenomenon where neurons in a layer learn to rely on each other too much, instead of learning independently from the input.

The dropout rate is a hyperparameter that controls the percentage of neurons that will be dropped out. A higher dropout rate means that more neurons will be dropped out, which can help to reduce overfitting, but it also increases the risk of underfitting. A lower dropout rate means that fewer neurons will be dropped out, which can help to reduce underfitting, but it also increases the risk of overfitting.

During inference, dropout is typically not applied, so all neurons are used to make predictions.

Dropout is a powerful regularization technique that can improve the performance of deep neural networks, by reducing the co-adaptation of the neurons and preventing overfitting. It should be applied after fully connected layers and also sometimes to the last layer of the convolutional neural networks, depending on the architecture.

---

### 15. When do we introduce DropOut, or when do we know we have some overfitting

Dropout is a regularization technique that is typically introduced when overfitting is detected in a deep neural network. Overfitting occurs when a model performs well on the training data but poorly on the validation or test data. This happens when a model becomes too complex and starts to memorize the noise or random variations in the training data, rather than learning the underlying patterns.

There are several signs that can indicate overfitting, such as:
1. The training loss is low, but the validation loss is high.
2. The training accuracy is high, but the validation accuracy is low.
3. The model performs well on the training data but poorly on the test data.
4. The model is not generalizing well to new, unseen data.
	
Once overfitting is detected, dropout can be introduced to the model as a regularization technique to help prevent overfitting and improve the performance on the validation and test data. It can be applied after fully connected layers and also sometimes to the last layer of the convolutional neural networks, depending on the architecture.

Dropout is typically introduced with a low rate, such as 0.2 or 0.3 and then the rate can be increased if overfitting is still present. Also, it is important to monitor the accuracy and loss of the validation set during training to make sure that the dropout is not causing underfitting

---

### 16. The distance of MaxPooling from Prediction:
In a convolutional neural network (CNN), the distance of the max pooling layer from the prediction refers to the number of layers between the last max pooling layer and the final fully connected layer (also known as the output layer) that produces the prediction.

The placement of the max pooling layer relative to the prediction can have an impact on the performance of the network. The closer the max pooling layer is to the prediction, the more abstract and complex features the network has to work with. However, this can also increase the number of parameters in the model and the computational complexity.

In general, the max pooling layer is placed closer to the beginning of the network, to reduce the spatial dimensionality of the feature maps, reduce the computational cost, and also make the features more robust to small translations of the input. As the input image is propagated through the network, the receptive field of a neuron increases, allowing the neuron to "see" more of the input image.

It's also important to note that in some architectures, like DenseNet and ResNet, the max pooling layer is not used, instead, they use the convolutional layers with stride 2 in order to reduce the spatial resolution.

---

### 17. The distance of Batch Normalization from Prediction:
In a convolutional neural network (CNN), the distance of the batch normalization layer from the prediction refers to the number of layers between the last batch normalization layer and the final fully connected layer (also known as the output layer) that produces the prediction.

The placement of the batch normalization layer relative to the prediction can have an impact on the performance of the network, however, the exact impact may vary depending on the architecture and the specific task.

In general, batch normalization is placed immediately after the convolutional layers or fully connected layers. This is because, the batch normalization normalizes the activations of a layer, making the training more stable and faster. It also helps to reduce the internal covariate shift, which is the change in the distribution of the inputs to a layer caused by the change in the parameters during training.

It's also important to note that in some architectures, batch normalization is not used, or it's used only on specific layers, depending on the architecture and the specific task.

---

### 18. When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)
In a convolutional neural network (CNN), the decision to stop using convolutional layers and move on to a larger kernel or other alternative depends on the specific task, the architecture of the network, and the performance of the model on the validation set.

One indication to stop using convolutional layers and move on to a larger kernel or other alternative is when the model has reached a satisfactory level of performance on the validation set. This means that the model is generalizing well to new, unseen data and is not overfitting or underfitting.

Another indication to stop using convolutional layers and move on to a larger kernel or other alternative is when the model is becoming too complex and the computational cost is becoming too high. In this case, using a larger kernel or other alternative can help to reduce the number of parameters and the computational cost, while still preserving the important features learned by the network.

Additionally, one can also stop using convolutional layers and move on to a larger kernel or other alternative when the model is not able to extract more complex and abstract features from the input. In this case, using a larger kernel or other alternative can help to extract more complex and abstract features from the input, which can improve the performance of the model.

---

### 19. How do we know our network is not going well, comparatively, very early
In a convolutional neural network (CNN), there are several signs that can indicate that the network is not performing well, comparatively, very early in the training process. These signs can help to identify problems with the model, the data, or the training process, and take corrective actions before the training process becomes too computationally expensive.

Some common signs that the network is not performing well, comparatively, very early in the training process are:

1. High training loss and low validation loss: This can indicate that the model is overfitting and memorizing the noise in the training data, rather than learning the underlying patterns.
2. Low training accuracy and low validation accuracy: This can indicate that the model is not learning from the data or that the data is too difficult for the model.
3. No improvement in the validation loss or accuracy after several training iterations: This can indicate that the model is not learning from the data or that the learning rate is too low.
4. Gradient explosion or vanishing: This can indicate that the model is too complex or that the learning rate is too high.
	
It's important to monitor the performance of the model on the validation set, early on during the training process, to detect these signs early, and take corrective actions, such as reducing the complexity of the model, increasing the regularization, changing the learning rate, or changing the data preprocessing.

---

### 20. Batch Size, and effects of batch size

Batch size is a hyperparameter in a deep neural network that controls the number of samples used in one forward/backward pass. The batch size can have an impact on the performance of the network, and the choice of batch size can depend on the specific task and the computational resources available.

A smaller batch size means that the model updates its parameters more frequently, but it also increases the variance of the gradients, which can make the training process less stable. A larger batch size means that the model updates its parameters less frequently, but it also decreases the variance of the gradients, which can make the training process more stable.

Here are the effects of different batch sizes:
	1. Large batch size: can result in faster convergence, but it can also cause the model to get stuck in a suboptimal solution.
	2. Small batch size: can result in slower convergence, but it can also help the model to escape local optima and converge to a better solution.
	
It's important to note that the batch size also affects the memory requirements of the model, a larger batch size requires more memory to store the intermediate results of the forward and backward pass.