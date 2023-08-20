********
# UNSUPERVISED LEARNING
*********

# K-Means Clustering

It is a centroid-based algorithm, where each cluster is associated with a centroid. The main aim of this algorithm is to minimize the sum of distances between the data point and their corresponding clusters.

The algorithm takes the unlabeled dataset as input, divides the dataset into k-number of clusters, and repeats the process until it does not find the best clusters. The value of k should be predetermined in this algorithm.

**The working of the K-Means algorithm is explained in the below steps:**

Step-1: Select the number K to decide the number of clusters.

Step-2: Select random K points or centroids. (It can be other from the input dataset).

Step-3: Assign each data point to their closest centroid, which will form the predefined K clusters.

Step-4: Calculate the variance and place a new centroid of each cluster.

Step-5: Repeat the third steps, which means reassign each datapoint to the new closest centroid of each cluster.

Step-6: If any reassignment occurs, then go to step-4 else go to FINISH.

Step-7: The model is ready.

    import numpy as np
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11], [4, 5]])
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
   
    colors = ['g.', 'r.']
     for i in range(len(X)):
        plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='p', s=150, linewidths=5, zorder=10)
    plt.show()


**Elbow Method**

The Elbow method is one of the most popular ways to find the optimal number of clusters. This method uses the concept of WCSS value. **WCSS stands for Within Cluster Sum of Squares**, which defines the total variations within a cluster. The formula to calculate the value of WCSS (for 3 clusters) is given below:

WCSS= ∑Pi in Cluster1 distance(Pi C1)2 +∑Pi in Cluster2distance(Pi C2)2+∑Pi in CLuster3 distance(Pi C3)2

To find the optimal value of clusters, the elbow method follows the below steps:

It executes the K-means clustering on a given dataset for different K values (ranges from 1-10).
For each value of K, calculates the WCSS value.
Plots a curve between calculated WCSS values and the number of clusters K.
The sharp point of bend or a point of the plot looks like an arm, then that point is considered as the best value of K.
Since the graph shows the sharp bend, which looks like an elbow, hence it is known as the elbow method. The graph for the elbow method looks like the below image:

![k-means-clustering-algorithm-in-machine-learning13](https://github.com/TirthGada/MachineLearning_practice/assets/118129263/e3cdba43-68d0-4e20-92fa-8aa375947817)


      #finding optimal number of clusters using the elbow method  
     from sklearn.cluster import KMeans  
     wcss_list= []  #Initializing the list for the values of WCSS  
  
     #Using for loop for iterations from 1 to 10.  
    for i in range(1, 11):  
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42)  
    kmeans.fit(x)  
    wcss_list.append(kmeans.inertia_)  
    mtp.plot(range(1, 11), wcss_list)  
    mtp.title('The Elobw Method Graph')  
    mtp.xlabel('Number of clusters(k)')  
    mtp.ylabel('wcss_list')  
    mtp.show()  

**K-Means: Inertia**
Inertia measures how well a dataset was clustered by K-Means. It is calculated by measuring the distance between each data point and its centroid, squaring this distance, and summing these squares across one cluster. A good model is one with low inertia AND a low number of clusters ( K ).


*******
# Silhouette Coefficient:

Silhouette Coefficient or silhouette score is a metric used to calculate the goodness of a clustering technique. Its value ranges from -1 to 1.

1: Means clusters are well apart from each other and clearly distinguished.

0: Means clusters are indifferent, or we can say that the distance between clusters is not significant.

-1: Means clusters are assigned in the wrong way.

**Silhouette Score = (b-a)/max(a,b)**
where
**a= average intra-cluster distance i.e the average distance between each point within a cluster.**
**b= average inter-cluster distance i.e the average distance between all clusters.**

      KMean= KMeans(n_clusters=2)
      KMean.fit(Z)
      label=KMean.predict(Z)
      print(f'Silhouette Score(n=2): {silhouette_score(Z, label)}')
      Output: Silhouette Score(n=2): 0.8062146115881652

      
![1_cUcY9jSBHFMqCmX-fp8BvQ](https://github.com/TirthGada/MachineLearning_practice/assets/118129263/fa26defe-41f7-4b03-8849-7059f12084b8)

      
      sns.scatterplot(Z[0],Z[1],hue=label)

We can also use the silhouette score to check the optimal number of clusters. 
*********
# Hierarchical Clustering 

In this algorithm, we develop the hierarchy of clusters in the form of a tree, and this tree-shaped structure is known as the **dendrogram.**

Sometimes the results of K-means clustering and hierarchical clustering may look similar, but they both differ depending on how they work. As there is no requirement to predetermine the number of clusters as we did in the K-Means algorithm.

The hierarchical clustering technique has two approaches:

**Agglomerative**: Agglomerative is a bottom-up approach, in which the algorithm starts with taking all data points as single clusters and merging them until one cluster is left.

**Divisive**: Divisive algorithm is the reverse of the agglomerative algorithm as it is a top-down approach.


## Why hierarchical clustering?

As we already have other clustering algorithms such as K-Means Clustering, then why we need hierarchical clustering? So, as we have seen in the K-means clustering that there are some challenges with this algorithm, which are a predetermined number of clusters, and it always tries to create the clusters of the same size. To solve these two challenges, we can opt for the hierarchical clustering algorithm because, in this algorithm, we don't need to have knowledge about the predefined number of clusters.

## Agglomerative Hierarchical clustering

The agglomerative hierarchical clustering algorithm is a popular example of HCA. To group the datasets into clusters, it follows the bottom-up approach. It means, this algorithm considers each dataset as a single cluster at the beginning, and then start combining the closest pair of clusters together. It does this until all the clusters are merged into a single cluster that contains all the datasets.

### How the Agglomerative Hierarchical clustering Work?

The working of the AHC algorithm can be explained using the below steps:

Step-1: Create each data point as a single cluster. Let's say there are N data points, so the number of clusters will also be N.
How the Agglomerative Hierarchical clustering Work?

Step-2: Take two closest data points or clusters and merge them to form one cluster. So, there will now be N-1 clusters.

Step-3: Again, take the two closest clusters and merge them together to form one cluster. There will be N-2 clusters.

Step-5: Once all the clusters are combined into one big cluster, develop the dendrogram to divide the clusters as per the problem.

### Woking of Dendrogram in Hierarchical clustering

The dendrogram is a tree-like structure that is mainly used to store each step as a memory that the HC algorithm performs. In the dendrogram plot, the Y-axis shows the Euclidean distances between the data points, and the x-axis shows all the data points of the given dataset.

The working of the dendrogram can be explained using the below diagram:
![hierarchical-clustering-in-machine-learning10-2](https://github.com/TirthGada/MachineLearning_practice/assets/118129263/024f5e5d-6564-4a43-a090-3e3974b184f0)

**Ward linkage**: Also known as MISSQ (Minimal Increase of Sum-of-Squares). It specifies the distance between two clusters, computes the sum of squares error (ESS), and successively chooses the next clusters based on the smaller ESS. Ward's Method seeks to minimize the increase of ESS at each step. Therefore, minimizing error.

     from sklearn.cluster import AgglomerativeClustering

     clustering_model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
     clustering_model.fit(selected_data)
     clustering_model.labels_
     data_labels = clustering_model.labels_
     sns.scatterplot(x='Annual Income (k$)', 
                y='Spending Score (1-100)', 
                data=selected_data, 
                hue=data_labels,
                palette="rainbow").set_title('Labeled Customer Data')

**
**IF DATASET IS SMALL GO WITH HIERARCHIAL CLUSTERING**
**
### Dendogram
     import scipy.cluster.hierarchy as shc
     import matplotlib.pyplot as plt

     plt.figure(figsize=(10, 7))
     plt.title("Customers Dendrogram")

     # Selecting Annual Income and Spending Scores by index
     selected_data = customer_data_oh.iloc[:, 1:3]
     clusters = shc.linkage(selected_data, 
            method='ward', 
            metric="euclidean")
            shc.dendrogram(Z=clusters)
            plt.show()


![definitive-guide-to-hierarchical-clustering-with-python-and-scikit-learn-10](https://github.com/TirthGada/MachineLearning_practice/assets/118129263/3ea34b5f-3e80-4f57-8360-58fd58911c4b)


# Density-Based Spatial Clustering Of Applications With Noise (DBSCAN)

The DBSCAN algorithm is based on this intuitive notion of “clusters” and “noise”. The key idea is that for each point of a cluster, the neighborhood of a given radius has to contain at least a minimum number of points. 

### Why DBSCAN? 
Partitioning methods (K-means, PAM clustering) and hierarchical clustering work for finding spherical-shaped clusters or convex clusters. In other words, they are suitable only for compact and well-separated clusters. Moreover, they are also severely affected by the presence of noise and outliers in the data.
