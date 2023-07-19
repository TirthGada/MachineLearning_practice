# MachineLearning_practice

**1)Diabetes dataset**:-Trained dataset using **linear regression** and plotted the trained and tested data points using **Numpy** , **Scikit** , **Matplotlib** .


**2)Medical charges incurred dataset**:-Trained dataset using **linear regression** and plotted the trained and tested data points using **Numpy** , **Scikit** , **Matplotlib** ,**Seaborn**. Plotted various graph of i)BMI vs Med-charges  ii)Age vs Med-charges . Also computed loss function which gave a great insight that medical charges incurred by a person depends on various factors such as **age , gender , region , smoker/non-smoker,BMI , no of childrens** . Created a barplot using **Seaborn** which showed that usually smokers incur **60-65%** higher medical charges as compared to non-smokers.

**3)Iris dataset**:- Trained dataset with **Scikit** using **KNeighbors Classification** which takes features of flowers such as **sepal_width, sepal length,petal length,petal width** and predicts whethers the folwer belongs to **Iris-Setosa, Iris-Virginica** etc..

**4)Housing price dataset**:- Used various regressor like **RandomForest , Multi-Linear ,DecisionTree** and   **KNeighbors Classification** using **Scikit**  etc.. Plotted various histograms depicting correlation with price of house with various parameters like carpet-area, guestroom , locality , bedrooms, bathroo etc.. using **Pyplot** . Source of dataset was **Kaggle**

**5)ICICI BANK Stock prediction**:- Predicted stock prices using **multiple linear regression**


*******
## Decision Tree Classifier
**1)INTRODUCTION**: Decision Tree is a Supervised learning technique that can be used for both classification and Regression problems, but mostly it is preferred for solving Classification problems. It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome.

In a Decision tree, there are two nodes, which are the Decision Node and Leaf Node. Decision nodes are used to make any decision and have multiple branches, whereas Leaf nodes are the output of those decisions and do not contain any further branches.
The decisions or the test are performed on the basis of features of the given dataset.

It is a graphical representation for getting all the possible solutions to a problem/decision based on given conditions.
It is called a decision tree because, similar to a tree, it starts with the root node, which expands on further branches and constructs a tree-like structure.

In order to build a tree, we use the CART algorithm, which stands for Classification and Regression Tree algorithm.
A decision tree simply asks a question, and based on the answer (Yes/No), it further split the tree into subtrees.

**2)Decision Tree Terminologies**

Root Node: Root node is from where the decision tree starts. It represents the entire dataset, which further gets divided into two or more homogeneous sets.

Leaf Node: Leaf nodes are the final output node, and the tree cannot be segregated further after getting a leaf node.

Splitting: Splitting is the process of dividing the decision node/root node into sub-nodes according to the given conditions.

Branch/Sub Tree: A tree formed by splitting the tree.

Pruning: Pruning is the process of removing the unwanted branches from the tree.

Parent/Child node: The root node of the tree is called the parent node, and other nodes are called the child nodes.

**3)Process**

Step-1: Begin the tree with the root node, says S, which contains the complete dataset.

Step-2: Find the best attribute in the dataset using Attribute Selection Measure (ASM).

Step-3: Divide the S into subsets that contains possible values for the best attributes.

Step-4: Generate the decision tree node, which contains the best attribute.

Step-5: Recursively make new decision trees using the subsets of the dataset created in step -3. Continue this process until a stage is reached where you cannot further classify the nodes and called the final node as a leaf node.


********

## RANDOM FOREST ALGORITHM

**from sklearn.ensemble import RandomForestClassifier** 

**1)INTRODUCTION**:- Random Forest is a popular machine learning algorithm that belongs to the supervised learning technique. It can be used for both Classification and Regression problems in ML. It is based on the concept of ensemble learning, which is a process of combining multiple classifiers to solve a complex problem and to improve the performance of the model.

As the name suggests, "Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset." Instead of relying on one decision tree, the random forest takes the prediction from each tree and based on the majority votes of predictions, and it predicts the final output.Although random forest can be used for both classification and regression tasks, it is not more suitable for Regression tasks.

**Below are some points that explain why we should use the Random Forest algorithm:**

It takes less training time as compared to other algorithms.

It predicts output with high accuracy, even for the large dataset it runs efficiently.

It can also maintain accuracy when a large proportion of data is missing.

Random Forest works in two-phase first is to create the random forest by combining N decision tree, and second is to make predictions for each tree created in the first phase.

**The Working process can be explained in the below steps and diagram:**

Step-1: Select random K data points from the training set.

Step-2: Build the decision trees associated with the selected data points (Subsets).

Step-3: Choose the number N for decision trees that you want to build.

Step-4: Repeat Step 1 & 2.

**Applications of Random Forest**

There are mainly four sectors where Random forest mostly used:

Banking: Banking sector mostly uses this algorithm for the identification of loan risk.

Medicine: With the help of this algorithm, disease trends and risks of the disease can be identified.

Land Use: We can identify the areas of similar land use by this algorithm.

Marketing: Marketing trends can be identified using this algorithm.

******
## K-Nearest Neighbor(KNN) Algorithm
**1)INTRODUCTION:**
K-Nearest Neighbour is one of the simplest Machine Learning algorithms based on Supervised Learning technique.

K-NN algorithm assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categories.

K-NN algorithm stores all the available data and classifies a new data point based on the similarity. This means when new data appears then it can be easily classified into a well suite category by using K- NN algorithm.

K-NN algorithm can be used for Regression as well as for Classification but mostly it is used for the Classification problems.
K-NN is a non-parametric algorithm, which means it does not make any assumption on underlying data.

It is also called a lazy learner algorithm because it does not learn from the training set immediately instead it stores the dataset and at the time of classification, it performs an action on the dataset.

KNN algorithm at the training phase just stores the dataset and when it gets new data, then it classifies that data into a category that is much similar to the new data.

**2)PROCESS**

Step-1: Select the number K of the neighbors

Step-2: Calculate the Euclidean distance of K number of neighbors

Step-3: Take the K nearest neighbors as per the calculated Euclidean distance.

Step-4: Among these k neighbors, count the number of the data points in each category.

Step-5: Assign the new data points to that category for which the number of the neighbor is maximum.

Step-6: Our model is ready.

**3)Advantages of KNN Algorithm:**

It is simple to implement.

It is robust to the noisy training data

It can be more effective if the training data is large.

**4)Disadvantages of KNN Algorithm:**

Always needs to determine the value of K which may be complex some time.

The computation cost is high because of calculating the distance between the data points for all the training samples.

********
## Multiple Linear Regression

In the previous topic, we have learned about Simple Linear Regression, where a single Independent/Predictor(X) variable is used to model the response variable (Y). But there may be various cases in which the response variable is affected by more than one predictor variable; for such cases, the Multiple Linear Regression algorithm is used.

********
## Polynomial Regression

It is also called the special case of Multiple Linear Regression in ML. Because we add some polynomial terms to the Multiple Linear regression equation to convert it into Polynomial Regression.

It is a linear model with some modification in order to increase the accuracy.

The dataset used in Polynomial regression for training is of non-linear nature.

It makes use of a linear regression model to fit the complicated and non-linear functions and datasets.
Hence, "In Polynomial regression, the original features are converted into Polynomial features of required degree (2,3,..,n) and then modeled using a linear model.

**y= b0+b1x1+ b2x12+ b2x13+...... bnx1n**

Simple Linear Regression equation:         y = b0+b1x         

Multiple Linear Regression equation:         y= b0+b1x+ b2x2+ b3x3+....+ bnxn        

Polynomial Regression equation:         y= b0+b1x + b2x2+ b3x3+....+ bnxn         
