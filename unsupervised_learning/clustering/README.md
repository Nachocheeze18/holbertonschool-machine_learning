A multimodal distribution has multiple peaks or modes in its data distribution.
A cluster is a group of data points that are similar or close to each other.
Cluster analysis is a statistical technique to group similar data points into clusters.
"Soft" clustering assigns data points to multiple clusters with probabilities, while "hard" clustering assigns each point to a single cluster.
K-means clustering is an algorithm that partitions data into K clusters based on their similarity.
Mixture models are probabilistic models that represent data as a combination of multiple probability distributions.
A Gaussian Mixture Model (GMM) is a mixture model that uses Gaussian distributions.
The Expectation-Maximization (EM) algorithm is used to estimate the parameters of statistical models, like GMMs.
To implement EM for GMMs, iteratively update means, covariances, and weights based on data likelihood.
Cluster variance measures the spread or dispersion of data points within a cluster.
The mountain/elbow method helps determine the optimal number of clusters by identifying an "elbow" point in the cluster variance plot.
The Bayesian Information Criterion is a metric used to select the best-fitting model, often applied in choosing the number of clusters.
Determine the correct number of clusters by comparing BIC or other metrics for different cluster counts.
Hierarchical clustering builds a tree-like structure of nested clusters.
Agglomerative clustering starts with individual data points and merges them into clusters.
Ward's method is an agglomerative clustering linkage criterion that minimizes variance when merging clusters.
Cophenetic distance measures the similarity between clusters in hierarchical clustering.
Scikit-learn is a Python library for machine learning and data mining.
Scipy is a scientific computing library in Python, often used in conjunction with scikit-learn for various data analysis tasks.