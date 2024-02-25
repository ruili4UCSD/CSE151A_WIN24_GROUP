# CSE151A_WIN24_GROUP

# LeetCode Problem Difficulty Prediction

## Introduction

This project aims to analyze and predict the difficulty levels of problems on LeetCode, a popular platform for interview coding preparation. By applying machine learning techniques to the LeetCode dataset, our aim is to uncover patterns that can help predict whether a problem is classified as Easy, Medium, or Hard.

## Dataset Overview

The dataset comprises various attributes of LeetCode problems, such as acceptance rate, difficulty level, and tags related to data structures and algorithms. Our initial task involved cleaning the data, handling missing values, and encoding categorical features to prepare the dataset for analysis and modeling.

[🔗 LeetCode Dataset](https://www.kaggle.com/datasets/gzipchrist/leetcode-problem-dataset/data)

## Data Preprocessing

We performed several preprocessing steps, including:

- Removing irrelevant features.
- Handling missing values through imputation or exclusion.
- Encoding categorical variables to numerical values.
- Normalizing the data using MinMax Scaling to ensure compatibility with machine learning algorithms.

## Exploratory Data Analysis (EDA)

Our EDA focused on gaining insights into the distribution of problem difficulties and identifying correlations between features. Key highlights include:

- Distribution plots to visualize the frequency of each difficulty level.
- Correlation analysis to understand the relationships between features, particularly how they relate to problem difficulty.
- Heatmaps to visually represent the correlation matrix and help identify significant predictors.

## Modeling

### K-Nearest Neighbors (KNN)

We implemented the K-Nearest Neighbors algorithm as our initial model, chosen for its simplicity and effectiveness in multi-label classification tasks. The model's key aspects included:

- Utilizing a range of 'k' values to determine the optimal number of neighbors for classification.
- Evaluating the model using accuracy, precision, and recall metrics.
- Employing cross-validation to assess the model's performance and avoid overfitting.

#### KNN Model Results:

The KNN model provided a baseline performance with varying results across different 'k' values. The optimal 'k' value of 12 was determined based on cross-validation scores, balancing the trade-off between bias and variance. The model's performance indicated potential areas for improvement, leading us to consider more complex algorithms for future work.

## Future Work

In the future, we plan to explore the following models:

- **Decision Tree**: We aim to leverage the interpretability of Decision Trees which will allow us to visualize the decision-making process and understand the importance of different features in predicting problem difficulty.
- **Neural Network**: Given the potential for high customization and optimization, we well implement a Neural Network to capture complex patterns in the data. We will experiment with different architectures, activation functions, and loss functions to optimize the model's performance.

[🔗 Current Notebook](./ipynb/LeetcodeDataExploration.ipynb)