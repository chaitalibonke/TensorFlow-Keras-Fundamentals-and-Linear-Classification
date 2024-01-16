#Choosing the Algorithms
Clearly stating your issue
AI books and blogs frequently discuss the various kinds of AI challenges that fall into the following categories:
• Supervised • Unsupervised • Semi-supervised • Reinforcement
When you first start working on a problem, you tend to focus more about what kind of problem you are trying to solve than what kind of problem it is.
Types of model problems include classification, regression, clustering, and anomaly detection.
What problem are we trying to solve?
• How important is it that we get this perfectly right?
• Do we value speed or accuracy?
• Can we get a decent answer now, and a better answer later?
A given problem type can be approached in a variety of ways, and there are even overlapping algorithms that can be applied to each distinct kind.
Support vector machines (SVMs), Random forests, Artificial Neural Networks (ANNs), K-Nearest Neighbors (KNN), and linear regression are some of the techniques used to understand regression difficulties.
Scikit-learn facilitates the process of breaking things down into manageable pieces.
Here are a few instances where there may have been wiser decisions, among many others:
• When there is no clear pattern to the relationship. You might wish to use a clustering algorithm if the items are clustered instead.
• If the data has a large number of outliers.
• If the association between your goal variable and your input or feature variables is not statistically significant. Using the temperature of the lunar surface to forecast the number of people wearing coats in Austin, Texas is unlikely to provide a useful model.
One method referred to as an ensemble technique is random forest. This basically means that various model outputs are integrated to generate a single output. The mean of the outputs is used in regression cases. This is akin to the wisdom of the crowds theory, which postulates that gathering responses from a large number of people will lead to superior results.
A decision tree is a straightforward sequence of true or false questions and responses that helps you arrive at a decision depending on how important a certain aspect is.
SVMs are among the most popular and too complex-sounding methods.
SVMs seek to divide the two closest categories into distinct groups while generating a line, plane, or hyperplane with the maximum distance between them.
The supports for this divider are the two points from the various categories that are closest to the dividing line; at that point, a vector that is a part of the main divider line is drawn; thus, the phrase "support vector." 
The margin is the separation between these two support vectors.
By using the kernel method, one can avoid the training costs associated with converting each data point to a higher dimension while still enjoying the advantages of working with high-degree polynomials.
This method, which is sometimes referred to as neural networks, is inspired by the way that neurons in the human brain make connections.
Neural networks are generally the final option to consider because of the hardware requirements and challenges in optimizing their performance.
Two or more layers make up neural networks, and each layer can concentrate on and modify the weights of a certain feature to get a better result than only one based on activator functions.
To ascertain if a neuron should be active or dormant, activator functions are only mathematical formulas.
Threshold Logic Units (TLUs) plus an input layer comprise the perceptron architecture, which is among the simplest ANN designs.

TLUs are neurons that use the inputs they receive to calculate a weighed total. In this instance, the activator function is known as a step function. When a prediction is made incorrectly during training, the weights of the other inputs are changed to reflect the correct response. Non-linear, complicated connection learning is not well suited for perceptrons.
• MLPs, or multilayer perceptrons, What surpasses a single layer? many layers. All an MLP does is build upon a perceptron base by adding extra layers to uncover more intricate correlations.
• Convolutional Neural Networks (CNNs): Often used in vision-related challenges, CNNs can examine regions of a video or image to obtain context rather than focusing on individual pixels.
• Deep Neural Networks (DNN): This is only a term used to indicate a network with numerous layers. This implies that in addition to the input and output layers, there are hidden layers as well. Theoretically, the number of layers you can have is infinite. Additionally, this is the context for the term Deep Learning (DL). DNNs have examples such as MLPs.
The point at which a decision is made is known as the decision boundary.
A second popular method for detecting anomalies is the use of isolation forests, which divide objects into those that are anomalous or not by creating a path along tree nodes. Because it believes anomalies will stand out and be simpler to understand, it considers that anything shorter in the tree and reaching the ending node faster is an abnormality.
Another ensemble strategy that makes use of the power of averages is isolation forests. An Isolation Tree (iTree) is the name given to each tree that is produced.
Finding parts of a dataset with enough comparable features to allow you to distinguish clearly between the individual points is known as clustering.
Types: K denotes clustering and DBScan
An unsupervised modeling method called Density-Based Spatial Clustering of Applications with Noise, or DBScan for short, looks for collections of high-density elements that are divided from other groups by areas of lower density.
Using an unsupervised approach called K-means clustering, all items in the datasets are grouped into k groups based on the smallest average distance between each point and the center.
With the advent of AutoML and other low- and no-code solutions that enable people with less technical expertise to generate models, the use of AI models is only going to grow in the future years. Although you encounter a number different models, you must be able to articulate the reasoning behind each one. 
It is challenging to eradicate bias once it has entered.
In order to obtain specific answers regarding how it arrived at its conclusion, we'll investigate our options for taking down the so-called "black box" surrounding it. 
Given our growing understanding of the factors that matter most in ensuring that patient and community health get the proper attention, this could be of utmost importance.
This chapter will address the following subjects: Recognizing the importance of interpretation 
investigating models that can be understood through design Interpreting model results using LIME Model outputs explained using SHAP.
Boston Dataset: How can I use SHAP to interpret the Boston Housing prediction?
The SHAP module is used to generate interpretable predictions using machine learning models. It allows us to observe which feature variables influence the predicted value. To put it another way, it can determine SHAP values, or the amount that a particular feature variable would raise or lower the projected variable.
A scalable machine learning method for tree boosting is called XGBoost, and it is an ensemble machine learning technique based on trees. To learn more about the parameters that make the algorithm function and when to utilize it, read on.
When Is XGBoost Useful?
when the training data contains a large number of observations.
Number of observations in training data < number of features.
When data has only numeric features or a combination of numerical and categorical features, it performs well.
when taking into account the model performance measures.



Tensorflow with Keras
Since the most common use of the gradient tape is to compute the gradient of a loss with respect to a list of trainable variables, trainable variables are observed by default.
The gradient tape is an incredibly useful tool that can even do calculations. 
Second-order gradients, that is to say, the gradient of a gradient.
Developing machine learning models with ease thanks to TensorFlow.
Build a linear classifier with the ability to distinguish between these two blobs. 
An affine transformation with training set to minimize the square of the difference between predictions and targets is called a linear classifier (prediction = W • input + b).
Show the training data points' classification by our linear model on a plot.
A given input point will be categorized as "0" if its forecast value is less than 0.5 and as "1" if it is more than 0.5. The targets are zeros and ones.
One or more tensors can be fed into and output from a layer, which is a data processing module. While some layers are stateless, most layers include a state that consists of the layer's weights and one or more tensors that were learned via stochastic gradient descent and collectively store the knowledge of the network.
Simple vector data, encoded in rank-2 tensors of shape (samples, features), is commonly handled by densely connected layers, sometimes termed fully connected or dense layers (the Dense class in Keras). 
Sequence data, recorded in rank-3 tensors of shape (samples, timesteps, features), is often processed using recurrent layers, such as an LSTM layer, or 1D convolution layers (Conv1D). 
2D convolution layers, or Conv2D, are typically used to process image data that is stored in rank-4 tensors.
An object called a Layer has some computation (a forward pass) and some state (weights). The computation is defined in the call() function, and the weights are usually defined in a build() (but they might also be constructed in the constructor, __init__()).
It is crucial to select the appropriate loss function for the given situation. Your network will take any possible shortcut to reduce loss, so if the goal isn't perfectly correlated with the task at hand's performance, it may wind up doing things you didn't want.
The data given will be iterated over in batches (of size batch_size) by evaluate(), which will return a list of scalars with the validation metrics appearing after the validation loss in the first item.
To perform inference more effectively, utilize the predict() function.
