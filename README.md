# CSE151A_WIN24_GROUP


# Milestone1 - Abstract

### Topic: LeetCode

**Dataset**:
https://www.kaggle.com/datasets/gzipchrist/leetcode-problem-dataset/data 

**Abstract**: 
Leetcode is one of the largest interview coding prep platforms on the internet, with a huge catalog of coding problems ranked easy, medium, and hard. However, the company has never explained what distinguishes these categories. The dataset we plan to analyze represents 1825 Leetcode problems, with 19 different features such as difficulty, acceptance rate, attempted frequency, and related topics for each individual question. The purpose of our project is to understand what factors are considered to be most important to Leetcode when scoring the difficulty of a problem. We first plan to conduct an exploratory analysis to gain insight into what features are most responsible for determining difficulty ratings. We will then build a machine learning algorithm that hopefully can accurately categorize the difficulty of a problem when given these attributes. Since the problem is a classification problem the models we are considering are k-nearest neighbors, decision trees, and a softmax neural network (if covered in class). One possible application of our project would be to automatically label the difficulty of coding problems encountered outside of Leetcode, such as HackerRank or Codesignal.


# Milestone2

**TODO**

https://docs.google.com/document/d/1iBJDfrBeEwS6ycuCgk8TkaNQDnpyRXVEpxZkiT6YH44/edit

Team 1:
- creating the github
- Describe the data in detail (evaluate data, # of observations, details about data distributions, scales, missing data, column descriptions)

Team 2:
- Data and information visualization
- - Correlation Coefficient Matrix = Heatmap (James)
- - Pairplot (Evie)
- - Scatterplots (David)
- Look at subsample of data
- - Features:
- - acceptance rate, accepted, and submissions

Team 3: 
- Preprocess data (encoding, standardization, normalization, transformations, imputations)
- - Swap difficulty to 0 1 and 2
- - drop: id, title, description, solution_link, url, asked_by_faang
- - change companies into faang count/total count
- - one-hot the related_topics
