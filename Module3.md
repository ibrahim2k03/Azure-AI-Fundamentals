# Module 3: Introduction to Machine Learning Concepts (1200 XP)
### Introduction
**Machine learning** is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It uses statistical techniques to enable machines to acquire knowledge from data and make predictions or decisions based on that knowledge.
### Machine learning models
**Machine learning models** are algorithms that can learn from data and make predictions or decisions based on that data.
1. Training data consists of past observations from features and their corresponding outcomes (labels). This observed data is used to teach the model.
2. An algorithm is applied to the data to identify patterns and relationships between the features (x) and the outcomes (y).
3. The result of the algorithm is a model that encapsulates the calculation dervied by the algorithm as a function. y=f(x)
4. After training, the model can be used for inferencing on new data to make predictions or decisions.
### Types of machine learning model
#### Supervised learning
training data consists of input features (x) and their corresponding correct outputs (y). Identify relationships between features and labels in past observations so that unknown labels can be predicted for new data.
- Regression: Label is predicted by model is numeric value
- Classification: Label is predicted by model is categorical value
#### Unsupervised learning
training data consists of input features (x) only. Identify patterns and relationships between the features (x) without knowing the correct outputs (y).
- Clustering: Group similar data points together based on their features.

### Regression
Regression models are trianed to predict numeric values based on input features. 
1. Split training data into training and validation sets.
2. Use algorithm to fit the training data to a model.
3. Use the validation data to test the model by predicting labels for the features
4. Compare the predicted labels with the actual labels to evaluate the model's performance.
#### Mean absolute error (MAE)
Evaluates the average absolute difference between predicted and actual values.
#### Mean squared error (MSE)
Evaluates the average squared difference between predicted and actual values. This is done to penalize larger errors more heavily.
#### Root mean squared error (RMSE)
Evaluates the square root of the average squared difference between predicted and actual values. This provides a measure of error in the same units as the target variable.
#### Coefficient of determination (R²)
Evaluates the proportion of the variance in the dependent variable that is predictable from the independent variable(s).
#### Iterative training
Training process that involves multiple cycles of learning and improvement with each cycle refining the model's parameters based on feedback from the data.
### Binary classification
Trained to predict binary outcomes (e.g., yes/no, true/false).
- We use an algorithm to fit the training data to a function that calculates probabilities for each class.
#### Confusion matrix 
A table used to describe the performance of a classification model on a set of test data for which the true values are known.
- True negative: The model correctly predicted the negative class. y^ = 0, y = 0
- True positive: The model correctly predicted the positive class. y^ = 1, y = 1
- False negative: The model incorrectly predicted the negative class. y^ = 0, y = 1
- False positive: The model incorrectly predicted the positive class. y^ = 1, y = 0

Correct predictions are on the diagonal, incorrect predictions are off the diagonal.
Accuracy = (True positive + True negative) / (Total predictions)
#### Recall
Proportion of positive cases that were correctly identified. 
- Recall = True positive / (True positive + False negative)
- How many actual positive cases were identified?
#### Precision
Proportion of predicted positive cases that were actually positive. 
- Precision = True positive / (True positive + False positive)
- How many predicted positive cases were actually positive?
#### F1 score
Harmonic mean of precision and recall. 
- F1 = 2 * (Precision * Recall) / (Precision + Recall)
#### Area under the ROC curve (AUC-ROC)
Measures the model's ability to distinguish between classes. Changes based on the threshold. 
- ROC (Receiver Operating Characteristic) curve: plots the true positive rate against the false positive rate at various threshold settings.
### Multiclass classification
Trained to predict categorical outcomes with more than two classes.
#### Training algorithm
##### One-vs-Rest (OvR)
- Train one binary classifier for each class against all other classes, each calcuilating the probability that the observation belongs to that class.
- The class with the highest probability is selected as the prediction.

### Clustering
Groups similar data points together based on their features. This doesn't require labeled data.
#### Training K-Means
1. feature (x) values are vectorized to define n-dimensional coordinate (for n features), we plot these coordinates on a graph
2. Decide how many clusters (k) we want to identify. The k points are randomly placed on the graph, these points are called centroids
3. Each data is assigned to its nearest centroid
4. The centroids are moved to the center of their assigned data points based mean distance
5. After the centeroid is moved, the data points may now be closer to a different centroid, so the data points are reassigned to clusters based on nearest centroid
6. The centeroid movement and cluster rellocation steps are repeated until the centroids no longer move significantly
#### Evaluation
No known label with  which to compare the clusters. Evaluation is based on how well the clusters are seperated from each other.
- **Average distance to cluster center**: How close on average each point in the cluster is to the centeroid of the cluster
- **Average distance to other cneter**: How close on abverge each point in the cluster is to the centeroid of all other clusters.
- **Maximum distance to cluster center**: Furthest distance between a point in the cluster to its centroid.
- **Silhouette score**: Value between -1 and 1 that summarizes ratio of ditstance between points in the same cluster and points in different clusers. Closer to 1 indicates better clustering.

### Deep learning
Emulate the way the human brain learns using artificial neural network that simulates electrochemical activity in biological neurons using mathemtical functions.
- Each neuron is a function that operates on inputs (x) and weight (w). The function is wrapped in an activation function that determines whether to pass the output on.
- Artificial neural networks are made up of multiple layers of neurons.
- Deep neural networks can be used for nlp and computer vision.
- When fitting the data to predict label y, the function f(x) isthe outer layer of the nested function in which each layer of the neural network encapsulates functions that operate on x and the weight w values associated with them
- The algorithm used to train the model invovles iteratively feeding feature values in the training data forward through the layers to calculate output values for y^, validating the model to evaluate how fasr off the predictions are from the actual values, and then adjusting the weights to minimize the error.
#### How neural networks learn
The weights in a neural network are central to how it calculates predicted values for labels. During the training process, the model learns the weights that will result in the most accurate predictions
1. Training and validation sets defined and training features are fed into the input layer
2. Neurons in each layer of the network apply their weights and feed the data through the network
3. The output layer produces a vector containing the calculated values.
4. The loss function compares the calculated values to the actual values to determine the error.
5. Since the entire network is one large nested function, optimzation fucntion can use differential calculus to evaluate tje influence of each weight in the network on the loss, and determine how they can be adjusted to reduce the loss. We can use gradient decent  in which each weight is increased or decreased to minimize loss.
6. The changes to the weight are backpropagated to the layers in the network, replacing the previously used values.
7. The process is repeated over multiple iterations (epochs) until the loss is minimize and the model predits acceptably accurately

### Summary
