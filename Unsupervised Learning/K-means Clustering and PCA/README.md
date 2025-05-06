# Airbnb Listing Segmentation with K-Means and PCA

This project applies **unsupervised machine learning** techniques to segment Airbnb listings in major European cities using **K-Means clustering** and **Principal Component Analysis (PCA)**. The goal is to uncover meaningful groupings in the data based on features like price, location, guest satisfaction, and room capacity.

---

## Overview

- **Dataset**: Cleaned Airbnb listings from Europe (sourced from Kaggle)
  https://www.kaggle.com/datasets/dipeshkhemani/airbnb-cleaned-europe-dataset
- **Methods**:
  - `K-Means` for clustering listings into similar groups
  - `PCA` to reduce dimensionality and enable 2D visualization
- **Goal**: Identify and interpret clusters of similar listings (e.g., budget, luxury, group stays)

---

## K-Means Clustering

**K-Means** is an unsupervised algorithm that partitions the data into `k` distinct clusters. It works by:

1. Randomly initializing `k` cluster centroids
2. Assigning each data point to the nearest centroid
3. Recomputing centroids as the mean of assigned points
4. Iterating until convergence

The objective is to minimize **intra-cluster variance**:

```
Minimize ∑ (i = 1 to k) ∑ (x in cluster_i) ||x - μ_i||^2
```

- Easy to implement and scale
- Assumes clusters are spherical and equally sized
- Sensitive to initialization and outliers

---

## Principal Component Analysis (PCA)

**PCA** is a linear technique for reducing the number of features while retaining as much variance as possible. It does so by:

1. Standardizing the data
2. Finding the eigenvectors (principal components) of the covariance matrix
3. Projecting the data onto the top components

Each component is a weighted combination of the original features:

```
PC_j = w_j1 * x_1 + w_j2 * x_2 + ... + w_jn * x_n
```

- **PC1** captures the most variance
- **PC2** is orthogonal and captures the next most
- Ideal for visualizing high-dimensional cluster structure

---

## Results

Listings were segmented into 4 interpretable clusters:
- Budget listings far from city center
- High-end urban listings with strong satisfaction
- Moderate listings balancing location and price
- Spacious group-friendly homes in suburban areas
