Eigendecomposition:

Eigendecomposition is a mathematical technique used to decompose a square matrix into a set of eigenvectors and eigenvalues. Given a matrix A, if there exist eigenvectors and eigenvalues such that A * v = λ * v, where v is the eigenvector and λ is the eigenvalue, then these eigenvectors and eigenvalues can be used to represent A in a factorized form. This is often used in various applications, including diagonalization of matrices and solving systems of linear differential equations.
Singular Value Decomposition (SVD):

Singular Value Decomposition is a matrix factorization technique that decomposes any given matrix into three other matrices: U, Σ (Sigma), and V^T (the transpose of V). It is widely used in data analysis, dimensionality reduction, and machine learning. SVD has various applications, such as principal component analysis (PCA), matrix approximation, and image compression.
Difference between Eig and SVD:

Eigendecomposition deals with square matrices and decomposes them into eigenvectors and eigenvalues. SVD, on the other hand, works for any m x n matrix and decomposes it into three matrices: U, Σ, and V^T. While both methods have applications in linear algebra and data analysis, SVD is more versatile and applicable to a wider range of matrices.
Dimensionality Reduction and Its Purposes:

Dimensionality reduction is the process of reducing the number of features (or dimensions) in a dataset while preserving as much of the important information as possible. Its purposes include:
Reducing computational complexity.
Removing noise and redundancy in data.
Visualizing high-dimensional data.
Improving model performance by reducing the curse of dimensionality.
Identifying the most important features.
Principal Components Analysis (PCA):

PCA is a linear dimensionality reduction technique used to transform data into a new coordinate system where the variance of the data is maximized along the axes (principal components). It identifies the principal components, which are linear combinations of the original features, to reduce the dimensionality of the data while preserving as much variance as possible.
t-Distributed Stochastic Neighbor Embedding (t-SNE):

t-SNE is a non-linear dimensionality reduction technique primarily used for visualization. It reduces the dimensionality of data while preserving the pairwise similarities between data points. t-SNE is effective at revealing clusters and patterns in high-dimensional data.
Manifold:

A manifold is a mathematical concept used in the context of dimensionality reduction. It is a topological space that locally resembles Euclidean space but may have a more complex global structure. In dimensionality reduction, techniques like t-SNE aim to preserve the underlying structure of data as if it were distributed on a lower-dimensional manifold.
Difference between Linear and Non-linear Dimensionality Reduction:

Linear dimensionality reduction techniques assume that the relationship between original features and reduced features is linear. They include methods like PCA and Linear Discriminant Analysis (LDA). Non-linear techniques, on the other hand, allow for more complex transformations and are better suited for capturing non-linear patterns in data. t-SNE is an example of a non-linear technique.
Techniques (Linear/Non-linear):

Linear Dimensionality Reduction Techniques: PCA, LDA, Linear Autoencoders.
Non-linear Dimensionality Reduction Techniques: t-SNE, Isomap, Locally Linear Embedding (LLE), Kernel PCA, Autoencoders with non-linear activation functions.