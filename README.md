# Drug-Activity-Prediction
Accuracy: 0.7742

Classification Algorithms: Random Forest, SVC, Naïve Bayes, Neural networks

Dimensionality Reduction: PCA, SVD

Approach:

I have followed two approaches:

1. The train set is divided into classes and data. Test set data and trainset data are combined. A dense matrix is generated with 100001 columns and 1150 rows. This is then passed to PCA , using sklearn libraries. This is tried with different number of components. The train set and test set are divided.

The classification algorithms, Neural networks and Naïve Bayes, Random Forest are used. This is tried with different parameters in random forest and neural networks.

2.The train set is divided into classes and data. Test set data and trainset data are combined. A CSR matrix is created. For dimensionality reduction, Truncated SVD is used. And for classification- SVM , neural networks, random forest are tried. CSR_idf is also tested. But the results did not vary much.

Conclusion:

The classification algorithms, SVC and naïve bayes ‘s performance is not even considerable. The random forest and neural networks produced results in the same range. Changing of the parameters in algorithms gave lesser F1 score always. With the above score, a sparse matrix with dimensionality reduction done with truncated SVD and classification with random forest is higher. 



