
Kurtosis Close to 3: A kurtosis value close to 3 suggests that the distribution of the target variable is similar to a normal distribution in terms of tailedness.

# MachineLearning_practice

*******
# SUPERVISED LEARNING
*******
### Residual 
Residual is the distance between predicted point (on the regression line) & actual point as depicted
******
# Support Vector Machine 
**1)INTRODUCTION**

Support Vector Machine or SVM is one of the most popular Supervised Learning algorithms, which is used for Classification as well as Regression problems. However, primarily, it is used for Classification problems in Machine Learning.The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane.

**2)Types**

SVM can be of two types:

**Linear SVM:** Linear SVM is used for linearly separable data, which means if a dataset can be classified into two classes by using a single straight line, then such data is termed as linearly separable data, and classifier is used called as Linear SVM classifier.

**Non-linear SVM:** Non-Linear SVM is used for non-linearly separated data, which means if a dataset cannot be classified by using a straight line, then such data is termed as non-linear data and classifier used is called as Non-linear SVM classifier.

SVM chooses the extreme points/vectors that help in creating the hyperplane. These extreme cases are called as support vectors, and hence algorithm is termed as Support Vector Machine.

Hence, the SVM algorithm helps to find the best line or decision boundary; this best boundary or region is called as a hyperplane. SVM algorithm finds the closest point of the lines from both the classes. These points are called support vectors. The distance between the vectors and the hyperplane is called as margin. And the goal of SVM is to maximize this margin. The hyperplane with maximum margin is called the optimal hyperplane.


*******
# Decision Tree Classifier
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

# RANDOM FOREST ALGORITHM

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

**Hyperparameter tuning**
https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/

******
# K-Nearest Neighbor(KNN) Algorithm
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

<img width="1440" alt="Screenshot 2023-10-13 at 4 39 04 PM" src="https://github.com/TirthGada/MachineLearning_practice/assets/118129263/9045f189-b0dd-4fa3-9e88-63a300b84693">
<img width="1440" alt="Screenshot 2023-10-13 at 4 38 55 PM" src="https://github.com/TirthGada/MachineLearning_practice/assets/118129263/06556a73-9371-4c66-b79a-eabe9d9d6bb1">


********
# Multiple Linear Regression

In the previous topic, we have learned about Simple Linear Regression, where a single Independent/Predictor(X) variable is used to model the response variable (Y). But there may be various cases in which the response variable is affected by more than one predictor variable; for such cases, the Multiple Linear Regression algorithm is used.

********
# Polynomial Regression

It is also called the special case of Multiple Linear Regression in ML. Because we add some polynomial terms to the Multiple Linear regression equation to convert it into Polynomial Regression.

It is a linear model with some modification in order to increase the accuracy.

The dataset used in Polynomial regression for training is of non-linear nature.

It makes use of a linear regression model to fit the complicated and non-linear functions and datasets.
Hence, "In Polynomial regression, the original features are converted into Polynomial features of required degree (2,3,..,n) and then modeled using a linear model.

Simple Linear Regression equation:         y = b0+b1x         

Multiple Linear Regression equation:         y= b0+b1x+ b2x2+ b3x3+....+ bnxn        

Polynomial Regression equation:         y= b0+b1x + b2x^2+ b3x^3+....+ bnx^n         
******
*********

## Regularization techniques
The key difference is in how they assign penalties to the coefficients:

Here are some considerations for both high and low regularization:

**High Regularization (Strong Penalty):**

Advantages:
Helps prevent overfitting: High regularization can be beneficial when you have a small dataset or when the features are highly correlated, as it reduces the risk of fitting noise in the data.
Simplifies the model: High regularization can lead to smaller coefficient values and a simpler model, which might be easier to interpret and generalize.
Feature selection: L1 regularization (Lasso) within Elastic Net can drive some coefficients to exactly zero, performing automatic feature selection.
Disadvantages:
Potential underfitting: Too much regularization can lead to underfitting, where the model is too simple to capture the underlying patterns in the data.
Loss of important information: Excessive regularization can lead to important features being penalized too heavily, potentially leading to a less accurate model.

**Low Regularization (Weak Penalty):**

Advantages:
Higher flexibility: Low regularization allows the model to fit the data more closely, potentially capturing complex relationships and patterns.
Better fit for large datasets: With a larger amount of data, the risk of overfitting might be reduced, making lower regularization more suitable.

**Key points**
Let us consider that we have a very accurate model, this model has a low error in predictions and it’s not from the target (which is represented by bull’s eye). This model has **low bias** and **low variance**. Now, if the predictions are scattered here and there then that is the symbol of **high variance**, also if the predictions are far from the target then that is the symbol of **high bias**.
Sometimes we need to choose between low variance and low bias. There is an approach that prefers some bias over high variance, this approach is called Regularization. It works well for most of the classification/regression problems.

**Ridge Regression:**
Performs L2 regularization, i.e., adds penalty equivalent to the square of the magnitude of coefficients
Minimization objective = LS Obj + α * (sum of square of coefficients)

**Lasso Regression:**
Performs L1 regularization, i.e., adds penalty equivalent to the absolute value of the magnitude of coefficients
Minimization objective = LS Obj + α * (sum of the absolute value of coefficients)

Here, LS Obj refers to the ‘least squares objective,’ i.e., the linear regression objective without regularization.

*******
# Lasso Regression

The full form of LASSO is the **least absolute shrinkage and selection operator**. As the name suggests, LASSO uses the “shrinkage” technique in which coefficients are determined, which get shrunk towards the central point as the mean.  

The LASSO regression in regularization is based on simple models that posses fewer parameters. We get a better interpretation of the models due to the shrinkage process. The shrinkage process also enables the identification of variables strongly associated with variables corresponding to the target. 

Lasso regression is also called **Penalized regression method**. This method is usually used in machine learning for the selection of the subset of variables. It provides greater prediction accuracy as compared to other regression models. Lasso Regularization helps to increase model interpretation. 

The less important features of a dataset are penalized by the lasso regression. The coefficients of this dataset are made zero leading to their elimination. The dataset with high dimensions and correlation is well suited for lasso regression. 

**Lasso Regression Formula:**

D= Residual Sum of Squares or Least Squares Lambda * Aggregate of  absolute values of coefficients   

**Process**
Here’s a step-by-step explanation of how LASSO regression works:

Linear regression model: LASSO regression starts with the standard linear regression model, which assumes a linear relationship between the independent variables (features) and the dependent variable (target). The linear regression equation can be represented as follows:y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε Where y is the dependent variable (target). β₀, β₁, β₂, ..., βₚ are the coefficients (parameters) to be estimated.x₁, x₂, ..., xₚ are the independent variables (features).ε represents the error term.

L1 regularization: LASSO regression introduces an additional penalty term based on the absolute values of the coefficients. The L1 regularization term is the sum of the absolute values of the coefficients multiplied by a tuning parameter λ: L₁ = λ * (|β₁| + |β₂| + ... + |βₚ|) Where:λ is the regularization parameter that controls the amount of regularization applied.β₁, β₂, ..., βₚ are the coefficients.

Objective function: The objective of LASSO regression is to find the values of the coefficients that minimize the sum of the squared differences between the predicted values and the actual values, while also minimizing the L1 regularization term:Minimize: RSS + L₁ Where:RSS is the residual sum of squares, which measures the error between the predicted values and the actual values.

Shrinking coefficients: By adding the L1 regularization term, LASSO regression can shrink the coefficients towards zero. When λ is sufficiently large, some coefficients are driven to exactly zero. This property of LASSO makes it useful for feature selection, as the variables with zero coefficients are effectively removed from the model.

Tuning parameter λ: The choice of the regularization parameter λ is crucial in LASSO regression. A larger λ value increases the amount of regularization, leading to more coefficients being pushed towards zero. Conversely, a smaller λ value reduces the regularization effect, allowing more variables to have non-zero coefficients.

Model fitting: To estimate the coefficients in LASSO regression, an optimization algorithm is used to minimize the objective function. Coordinate Descent is commonly employed, which iteratively updates each coefficient while holding the others fixed.

Helps to reduce overfitting by shrinking and selecting features with less importance.

**Video** 
https://www.youtube.com/watch?v=K8iKkzUDw5I

**Code**

    from sklearn.linear_model import Lasso
    #Initializing the Lasso Regressor with Normalization Factor as True
    lasso_reg = Lasso(normalize=True)
    #Fitting the Training data to the Lasso regressor
    lasso_reg.fit(X_train,Y_train)
    #Predicting for X_test
    y_pred_lass =lasso_reg.predict(X_test)
    #Printing the Score with RMLSE
    print("\n\nLasso SCORE : ", score(y_pred_lass, actual_cost))

**Example**

Let's assume we set λ = 0.5 for this example.

The Lasso regression aims to minimize the following cost function:
Lasso Cost = Least Squares Cost + 0.5 * (|β1| + |β2|)

Where β1 and β2 are the coefficients of the features X1 and X2, respectively.

Step 3: Model Training
Using the Lasso cost function, the model will find the coefficients (β1 and β2) that minimize the cost while considering the regularization term. The coefficients will be adjusted based on the data and the value of λ.

Step 4: Prediction
Once the model is trained, you can use it to make predictions. Given a new set of feature values (X1_new, X2_new), you calculate the predicted Y value as follows:

Predicted Y = β1 * X1_new + β2 * X2_new

However, in Lasso regression, due to the regularization term, some coefficients may become zero. This means that certain features might not contribute to the prediction at all.

In this example, if the Lasso regression determines that X2 is less important, it might set the coefficient β2 to zero. As a result, the prediction would only depend on the X1 feature:

Predicted Y = β1 * X1_new

This process results in a sparse model that can help in feature selection and model simplification.

********

# Ridge Regression

In Ridge regression, we add a penalty term which is equal to the square of the coefficient. The L2 term is equal to the square of the magnitude of the coefficients. We also add a coefficient  lambda  to control that penalty term. In this case if  \lambda  is zero then the equation is the basic OLS else

In ridge regression only difference is sum of absolute values are replaced with sum of squares.

**Video** https://www.youtube.com/watch?v=Yj7sIK0VMg0

*******

# Lasso vs Ridge 

**On applying
**Coefficient Shrinkage:**
Lasso: Lasso tends to shrink some coefficients all the way to zero, effectively performing automatic feature selection. It can be useful when you suspect that only a subset of the features are important for prediction.
Ridge: Ridge shrinks the coefficients towards zero but does not usually force them to become exactly zero. It can be effective when all features are potentially relevant and should be retained in the model.


**from sklearn.linear_model import Lasso, Ridge**

## Lasso Regression
lasso = Lasso(alpha=0.1)  # alpha is the regularization parameter
lasso.fit(X_train, y_train)
lasso_coeffs = lasso.coef_

## Ridge Regression
ridge = Ridge(alpha=1.0)  # alpha is the regularization parameter
ridge.fit(X_train, y_train)
ridge_coeffs = ridge.coef_
******
# General understanding
The penalty term is not added to the predicted outcome y itself in Lasso regression. The penalty term is only added to the optimization objective during the training process to influence the values of the coefficients (β) that are used to make predictions.

The primary goal of Lasso regression is to find the values of the coefficients that minimize the sum of squared residuals (||y - Xβ||^2) while also considering the L1 penalty term (λ||β||_1). The penalty term encourages some coefficients to become exactly zero, effectively performing feature selection and helping to prevent overfitting.


********
# Elastic Net regression

Elastic Net regression is a hybrid regularization technique that combines both Lasso (L1 regularization) and Ridge (L2 regularization) regression. It aims to address some of the limitations of Lasso and Ridge by providing a balance between feature selection and coefficient shrinkage. Elastic Net introduces two hyperparameters, α (alpha) and λ (lambda), that control the mix of L1 and L2 regularization.

The cost function of Elastic Net is a combination of L1 and L2 regularization terms along with the least squares cost:

**Elastic Net Cost = Least Squares Cost + α * λ * Σ|βi| + (1 - α) * λ * Σ(βi)^2**

Where:

Least Squares Cost: The traditional linear regression cost (sum of squared residuals).
α (alpha): A mixing parameter that determines the balance between L1 and L2 regularization. When α = 1, Elastic Net is equivalent to Lasso; when α = 0, it is equivalent to Ridge.
λ (lambda): The regularization parameter that controls the overall strength of regularization.


**Create an Elastic Net regression model**

alpha = 0.5  # Regularization strength
l1_ratio = 0.5  # L1/L2 ratio (0.5 means equal L1 and L2 regularization)
elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
elastic_net.fit(X_train, y_train)

**Guidelines**

If you set α (alpha) to 0.2 and λ (lambda) to 0.6 in the context of Elastic Net regularization, you would be specifying the following:

α = 0.2: This indicates that you want a mixture of both L1 and L2 regularization in your Elastic Net model, with a greater emphasis on L2 (Ridge) regularization. An α value of 0.2 means that 20% of the regularization is from L1 (Lasso) and 80% is from L2 (Ridge).
λ = 0.6: This represents the regularization strength or the penalty applied to the coefficients of the model. A higher value of λ leads to stronger regularization, which can shrink the coefficients and prevent overfitting. In this case, λ is set to 0.6, meaning that the regularization penalty is moderate.
By choosing α = 0.2 and λ = 0.6, you are prioritizing L2 regularization (Ridge) over L1 regularization (Lasso) and applying a moderate level of regularization penalty.

λ (lambda) = 8:
A higher value of λ indicates stronger regularization.
With λ = 8, the regularization effect on the coefficients will be significant, leading to potentially smaller coefficient values.
Larger λ values will push the coefficients towards zero more aggressively, resulting in a simpler model with fewer significant features.

********
# MOST IMPORTANT 

However, the magnitude of the coefficient alone doesn't necessarily determine the importance of a feature. Feature importance can also be affected by factors like the scale of the feature, its units, the distribution of the data, and potential interactions between features.

https://www.analyticsvidhya.com/blog/2020/11/lasso-regression-causes-sparsity-while-ridge-regression-doesnt-unfolding-the-math/

<img width="1440" alt="Screenshot 2023-08-10 at 12 43 56 AM" src="https://github.com/TirthGada/MachineLearning_practice/assets/118129263/6d04238d-2c3e-48c1-bfd0-9d4720ba7b4e">
<img width="1440" src="https://sparkbyexamples.com/wp-content/uploads/2023/03/Screenshot-2023-03-14-at-8.44.49-AM.png">


**********

# ROC-AUC CURVE

In classification, there are many different evaluation metrics. The most popular is accuracy, which measures how often the model is correct. This is a great metric because it is easy to understand and getting the most correct guesses is often desired. There are some cases where you might consider using another evaluation metric.

Another common metric is **AUC, area under the receiver operating characteristic (ROC) curve. The Reciever operating characteristic curve plots the true positive (TP) rate versus the false positive (FP) rate at different classification thresholds**.**The thresholds are different probability cutoffs that separate the two classes in binary classification**. It uses probability to tell us how well a model separates the classes.

![1*zs8w3RBo1RJcloz74Rm-zQ](https://github.com/TirthGada/MachineLearning_practice/assets/118129263/1eac33e2-c09f-4dd0-9c26-a6ad6de53005)

### Syntax


    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    print(f'model 1 AUC score: {roc_auc_score(y, y_proba)}')

###  Import points on ROC-AUC

An AUC score of around .5 would mean that the model is unable to make a distinction between the two classes and the curve would look like a line with a slope of 1. An AUC score closer to 1 means that the model has the ability to separate the two classes and the curve would come closer to the top left corner of the graph.


Because AUC is a metric that utilizes probabilities of the class predictions, we can be more confident in a model that has a higher AUC score than one with a lower score even if they have similar accuracies.



***********

# Cross Validation Score

Types :-

### K-Fold

The training data used in the model is split, into k number of smaller sets, to be used to validate the model. The model is then trained on k-1 folds of training set. The remaining fold is then used as a validation set to evaluate the model.


    from sklearn.model_selection import KFold, cross_val_score
    k_folds = KFold(n_splits = 5)
    scores = cross_val_score(clf, X, y, cv = k_folds)

### Stratified K-Fold

In cases where classes are imbalanced we need a way to account for the imbalance in both the train and validation sets. To do so we can stratify the target classes, meaning that both sets will have an equal proportion of all classes.
    
    sk_folds = StratifiedKFold(n_splits = 5)
    scores = cross_val_score(clf, X, y, cv = sk_folds)

### Leave-One-Out (LOO)

Instead of selecting the number of splits in the training data set like k-fold LeaveOneOut, utilize 1 observation to validate and n-1 observations to train. This method is an exaustive technique.

     loo = LeaveOneOut()
     scores = cross_val_score(clf, X, y, cv = loo)

### Leave-P-Out (LPO)
Leave-P-Out is simply a nuanced diffence to the Leave-One-Out idea, in that we can select the number of p to use in our validation set.

     lpo = LeavePOut(p=2)
     scores = cross_val_score(clf, X, y, cv = lpo)

*********

# Bootstrap Sampling

It uses technique of SRSWR (Simple random sampling with replacement . Very useful in case of small input dataset . In this even the same data point can be repeated in sample multiple times.

# Lazy vs Eager learner

**Eager learner** :- Model mainly completes its process in training phase itself

**Lazy learner** :- Model doesnt have training time , it does the work of predicting the output only when its told to predict rather than being trained earlier ( i.e it shows **laziness**)

