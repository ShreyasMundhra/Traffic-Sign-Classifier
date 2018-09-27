# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/class_examples.png "Class example images"
[image3]: ./examples/grayscale.png "Grayscaling"
[image4]: ./examples/aug.png "Original & Augmented Image"
[image5]: ./examples/sample_signs.png "Traffic Signs"
[image6]: ./examples/placeholder.png "Traffic Sign 2"
[image7]: ./examples/placeholder.png "Traffic Sign 3"
[image8]: ./examples/placeholder.png "Traffic Sign 4"
[image9]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed among the different classes in the training set, validation set and the test set.

![alt text][image1]

Here are example images corresponding to each of the 43 classes in the dataset.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because that will make it easier to train the network and reduce the required complexity due to smaller input size.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

As a last step, I normalized the image data because having zero mean and range from -1 to 1 for each input will reduce the number of epochs required to train the network.

I decided to generate additional data because training on additional images generated using different kinds of transformations might help the network to generalise better. I used random translation, random rotation and random scaling to add more data to the dataset. 

Here is an example of an original image and an augmented image:

![alt text][image4]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale normalised image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| 120 neurons        									|
| RELU					|												|
| dropout					| 0.5	for training											|
| Fully connected		| 84 neurons        									|
| RELU					|												|
| Fully connected		| 43 neurons        									|
| RELU					|												|
| Softmax				|        									|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer with a learning rate of 0.001 and a batch size of 256 and trained it for 40 epochs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 95.8%
* validation set accuracy of 93%
* test set accuracy of 91.9%

The Le-Net architecture was chosen along with a few small modifications as the network. The main modification was adding a dropout layer in the network. It is known to perform well on handwritten digits dataset MNIST. Hence, I decided to see if this architecture gives good results for traffic signs.

We can see that the model achieved a good validation and test accuracy. The train accuracy is higher than the validation and test accuracy as expected. However, it is not too high which means the model might have generalised well, because of the dropout layer that was added in the network. This provides evidence that the model is working well.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5]

I felt that most of the images in the dataset might have had a background with road, buildings and so on. Hence, I felt that the first image might be difficult to classify because a good part of the image has the sky as the background.

The second image has another red and white colored object behind it. The similarity in the color pattern might fool the network into thinking that it is part of the sign, which might make it difficult to classify this image.

The speck of dirt on the arrow and the white snow on the road made me feel that it might be difficult to classify the third image.

The fourth image has small cracks and signs of wear and tear in the traffic sign. The network might find it difficult to classify if it cannot handle texture well.

The fifth image, like the first image, has almost no road in the background, which might make it difficult for the network to classify if the network relies on the presence of a road in the background in order to identify traffic signs.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)      		| Speed limit (20km/h)   									| 
| No entry     			| No entry 										|
| Keep right					| Keep right											|
| Priority road	      		| Priority road					 				|
| Road work			| Right-of-way at the next intersection      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This was less than the test set accuracy of 91.9%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Note that the probabilities might not always exactly add upto 1 and might have minute deviations, due to inherent inaccuracies in floating point operations.

The top five soft max probabilities for the first image were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99867201e-01         			| Speed limit (20km/h)   									| 
| 6.72009191e-05     				| Speed limit (70km/h) 										|
| 6.38524361e-05					| Speed limit (30km/h)											|
| 9.67989422e-07	      			| Vehicles over 3.5 metric tons prohibited					 				|
| 6.56495956e-07				    | End of speed limit (80km/h)      							|

The top five soft max probabilities for the second image were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| No entry   									| 
| 1.04000475e-10     				| Stop 										|
| 4.92274589e-11					| Keep left											|
| 3.03031038e-12	      			| Turn left ahead					 				|
| 3.84393878e-13				    | Turn right ahead      							|

The top five soft max probabilities for the third image were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Keep right   									| 
| 3.31609439e-15     				| Roundabout mandatory 										|
| 1.02429630e-15					| Turn left ahead											|
| 3.26714318e-19	      			| Slippery road					 				|
| 3.09867641e-20				    | Double curve      							|

The top five soft max probabilities for the fourth image were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.91709888e-01         			| Priority road   									| 
| 7.49746431e-03     				| No vehicles 										|
| 5.34045277e-04					| Bicycles crossing											|
| 1.23890568e-04	      			| Yield					 				|
| 9.08984075e-05				    | Speed limit (50km/h)      							|

The top five soft max probabilities for the fifth image were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Right-of-way at the next intersection   									| 
| 1.17642540e-09     				| Beware of ice/snow 										|
| 4.70315735e-13					| Pedestrians											|
| 1.32530177e-13	      			| Dangerous curve to the right					 				|
| 1.72043352e-14				    | Double curve      							|

From the softmax probabilities for the five images, we notice that the model has at least 2 orders of magnitude more confidence in its prediction compared to the other predictions, which means that the model is very certain of its correct as well as incorrect predictions.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


