#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/tacticalhombre/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in cells 4, 5, 6, and 7 of the IPython notebook.  

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

I normalized the data and displayed samples. The code for this step is contained in cell 13 and 14 of the IPython notebook.  

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Fortunately, the data provided has already been split into training, validation, and testing.  So, no need to split the data. 

First, I created fake data by applying brightness adjustment, histogram equalization, or gamma adjustment.  Which one to perform is done randomly.  I took the class that has the max number of samples and use this number as a basis for adding data for each class.  If a classhas less than this number of samples, I keep creating samples for that class.  In the end, each of the 43 classes have equal number of images samples. Altogether, there are 51631 data points. Code for this is contained in cells 10-12.
 
#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 15th cell of the ipython notebook. 

My final model is based on the LeNet model with the addition of dropouts with a keep probability of 0.7 between FC layers. Below is the final model.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten  | Output 400
| Fully connected		| Output 120        									|
| RELU					|												|
| Dropout			| keep_prob 0.7        									|
| Fully connected		| Output 84        									|
| RELU					|												|
| Dropout			| keep_prob 0.7        									|
| Fully connected		| Output 43        									|

 


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 17th & 18th cell of the ipython notebook. 

Trained for 23 epochs, batch size 128, initial earning rate of 0.0005

Optimizer used is AdamOptimizer.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98.2
* validation set accuracy of 93.2 
* test set accuracy of 91.4

I used the LeNet model that was presented in the lessons.  With that model, I trained with the provided data but validation results did not cross over 90%.   

After adding normalization and dropouts, I was able to reach 93% validation accuracy.  I played around with dropout rates and settled for 0.07 as this value gave the best results.  Also experimented with different epochs but based on the loss curve gathered, convergence happens around 22-25 epochs.    


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Although all the 5 images searched from the web are clear, 3 of the 5 images are taken at an angle.  This may pose a difficulty for the model as not all of the training images have this orientation.

The results of predicting the 5 new traffic signs are provided in the Notebook.  The model only managed to make 1 correct prediction - 20% accuracy.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 24th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit (20 km/h)      		| Turn Left Ahead   									| 
| Right-of-way at next intersection     			| General Caution 										|
| Children Crossing				| Traffic Signals											|
| Roundabout Mandatory	      		| Roundabout Mandatory				 				|
| Ahead Only			| Turn Right Ahead      							|




#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were



| Image			        |     Prediction	(top)        					|  2nd Prediction | 3rd Prediction
|:---------------------:|:---------------------------------------------:|:-----------------------------------:| :-----------------------------------:|  
| 0 - Speed Limit (20 km/h)      		| 34 - Turn Left Ahead - 43.32%   									| Sign 18 - 20.48 %| Sign 17 - 8.35%|
| 11 - Right-of-way at next intersection     			| 18 - General Caution - 43.90%										| Sign 22 - 11.05%  | Sign 28 - 10.07%
| 28 - Children Crossing				| 26 - Traffic Signals	- 21.68%										| Sign 11 - 15.75%  | Sign 37 - 11.02%  |
| 40 - Roundabout Mandatory	      		| 40 - Roundabout Mandatory - 99.99%				 				|  Sign 37 - 0.01% | Sign 39 - 0.000017% |
| 35 - Ahead Only			| 33 - Turn Right Ahead  - 99.98%    							|  Sign 34 -  0.01%|  Sign 35 - 0.0096%|


