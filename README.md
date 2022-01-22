# Sparkify Project (Udacity Data Scientist Final).

The purpose of this project is to use log files from users to anticipate whether or not they would cancel their Sparkify account. The Sparkify service is a fictitious music service established by udacity to spoof the log files of actual music services like Spotify. A user can visit a variety of pages and engage in a variety of interactions, such as clicking Next Song, watching an advertisement, upgrading, or downgrading. The log files also contain user-specific information, such as the user's location and the user agent they are using to access the service.

<p align="left">
  <img alt="GitHub language count" src="https://img.shields.io/github/languages/count/DataScientist-GustavoAguiar/Disaster_Response_App?color=%2304D361">

  <img alt="Repository size" src="https://github-size-badge.herokuapp.com/DataScientist-GustavoAguiar/Disaster_Response_App.svg">

  <a href="https://rocketseat.com.br">
    <img alt="made by Gustavo Aguiar" src="https://img.shields.io/badge/made%20by-Gustavo-%237519C1">
  </a>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Table of Contents

- [Overview](#overview)
  - [Problem Statement](#problem_statement)
- [Project Structure](#structure)
  - [Load and Clean Dataset](#load_and_clean)
  - [Exploratory Data Analysis](#eda)
  - [Feature Engineering](#feature_engineering)
  - [Modeling](#modeling)
- [Conclusion](#conclusion)
  - [Metrics](#metrics)
  - [Results](#summary)
- [Files](#files)
- [Software Requirements](#sw)
- [Author](#author)
- [Credits and Acknowledgements](#credits)
- [License](#license)

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<a id='overview'></a>

## 1. Overview
To begin, we must define the project's goal and comprehend our business context. All of the decisions made in the following steps are consistent with our objectives.

Sparkify is a music streaming service. Many users use the Sparkify service on a daily basis, either using the free tier, which includes advertisements in between songs, or the premium subscription model, in which they stream music for free but pay a monthly flat rate. Users have the option to upgrade, downgrade, or cancel their service at any time.

Whatever the size of your company or your operational model, if you sell products or services, the problem is the same in almost every industry – customer retention analysis. Keeping your current customers with the company for as long as possible can be difficult. To keep your customers satisfied over time, you must understand their needs, provide excellent customer service, and understand why they would leave your company.

When you have this knowledge, you can significantly improve your retention rate and overall performance. One of the keys to preventing customer churn is to gather as much information about them as possible. That is the starting point, which is also required for machine learning predicting systems, which will assist you with the process of predicting customer churn.

<a id='problem_statement'></a>

## 1.1. Problem Statement

We have a JSON log of all actions taken by Sparkify users over a two-month period; our goal is to learn from this dataset which behaviors can help us predict whether users will "churn" (i.e. unsubscribe from the service). To accomplish this, we will extract the most relevant features from the log and train a machine learning classifier; in this article, we will work with a small subset, representing 1% of the total size, but we will use the Spark framework and keep scalability in mind, to ensure the same code can be reused when using the full dataset, which is 12GB in size.

This is a Customer Churn Prediction Problem , there are so many similar projects, such as [WSDM - KKBox's Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge) competition from [Kaggle](https://www.kaggle.com).

As a result, our job is to mine the customers' data and implement an appropriate model to predict customer churn in the following steps:

- Clean up the data: Analyze the NaN values, check the data types, and remove duplicates.
- EDA: Exploratory data analysis to examine feature distributions and correlations with key labels (churn).
- Feature engineering: Identifying and extracting customer features and customer behavior features to be considered in our model
- Train and measure models: I used Gradient Boost Tree classifier, Logistic Regression classifier, and Random Forest classifiers to train a baseline model and tune a better model from the best of them. It is worth noting that this data is unbalanced due to lower churn rates, so I chose 'f1 score' as a metric to assess model performance.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<a id='structure'></a>

## 2. Project Structure

There are four main sections in this project:

<a id='load_and_clean'></a>

### 2.1. Load and Clean Dataset

Load and clean the dataset, checking for invalid or missing data - for example, records without userids or sessionids.

<a id='eda'></a>

### 2.2. Exploratory Data Analysis (EDA)

Perform EDA by loading a small subset of the data and doing basic manipulations within Spark

<a id='feature_engineering'></a>

### 2.3. Feature Engineering

Once you've familiarized yourself with the data, build out the features you find promising to train your model on.

<a id='modeling'></a>

### 2.4. Modeling

Split the full dataset into train, test, and validation sets. Test out several of the machine learning methods you learned. Evaluate the accuracy of the various models, tuning parameters as necessary. Determine your winning model based on test accuracy and report results on the validation set.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<a id='conclusion'></a>

## 3. Conclusions

<a id='metrics'></a>

### 3.1. Metrics

For the machine learning model evaluation, selecting the appropriate metrics is critical. The issue is that the class of churned users is unbalanced due to the dataset's unequal distribution. In the situation of unequal classes, accuracy is misleading and should not be used. Instead, we'll employ precision-recall metrics. The precision indicates the percentage of churned users we successfully identified among all churned users, whereas the recall indicates the percentage of churned users we successfully identified among all churned users.

In most cases, the classes aren't distributed uniformly. There will always be a grey zone where classes are jumbled together since we cannot always clearly differentiate all points of a positive class from all points of a negative class. As a result, there is an inverse relationship between precision and recall: raising one parameter (precision or recall) decreases the other (recall or precision). The precision-recall tradeoff is what it's called. True positives are correctly identified churned users, false positives are non-churned users incorrectly identified as churned users, and false negatives are churned users incorrectly identified as non-churned users, and FN are false negatives are churned users incorrectly identified as non-churned users.

We want to establish a compromise between the precision and recall metrics in this case. Neither too many false positives nor too many false negatives are desirable. As a result, we'll utilize F1-score as our primary statistic, which is defined as harmonic mean of precision and recall; it combines the two metrics into a single metric that gives them equal weight - just what we need.

<a id='summary'></a>

### 3.2. Results

The initial full feature set was selected based on the general statistical method results and the previously defined delta threshold. Correlation tests between the selected features revealed that some of them had a significant relationship. As a result, some of the features from our initial set were removed.

I have tested three well-known ML models, which are: Random Forest, Gradient Boost Tree, and Logistic Regression with the same features selected after the correlation analysis (22). After that I selected the model which achieved better results considering the metrics assessed (_F1 = 0.9998_), and tried to optimize the already excellent results trough gridsearch and cross-validation. The best model was Gradient Boost Tree and the best hyperparameters were:

* maxIter=20
* maxDepth=5
* maxBins=32

The following were the model's five most important features:

* AvgSessionGap (~35%)
* AddFriendPerSessionHour (~10%)
* SessionCount (~9%)
* LogoutPerSessionHour (~7%)
* AddToPlaylistCount (~5%)
* ThumbsDownPerSessionHour (~5%)

Given the business context and the results presented in the EDA section, we can conclude (given our model's excellent results) that what most contributes to predicting a churning user is:

* **shorter time of inactivity between sessions**
* **fewer friend additions per session hour (i.e. less friends added)**
* **fewer sessions**
* **more logouts per session hour**
* **fewer songs were added to their playlists**
* **more interactions with thumbs down (i.e. dislike more frequently the songs played)**

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<a id='files'></a>

## 4. Files

<pre>
.
├── Sparkify_Final.ipynb ------------------------# MAIN PROJECT NOTEBOOK
├── README.md ------------------------# README FILE
├── LICENCE.md -------------------------------# LICENCE FILE
├── model_gbt_best -----------# SAVED PYSPARK ML MODEL

</pre>

<a id='sw'></a>

## 5. Software Requirements

This project uses **Python 3.6.3** and the necessary libraries are mentioned in _requirements.txt_.
The standard libraries which are not mentioned in _requirements.txt_ are _datetime_, _time_ and _math_.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<a id='author'></a>

## 6. Author

Gustavo Aguiar 👋🏽 Get in Touch!

Production Engineer | Master's Student in Production Engineering and Computational System | Data Analyst and Physical Simulation Engineering

[![Linkedin Badge](https://img.shields.io/badge/-Gustavo-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/gjmaguiar/?locale=en_US)](https://www.linkedin.com/in/gjmaguiar/?locale=en_US)
[![Gmail Badge](https://img.shields.io/badge/-gustavoaguiar@id.uff.br-c14438?style=flat-square&logo=Gmail&logoColor=white&link=mailto:gustavoaguiar@id.uff.br)](mailto:gustavoaguiar@id.uff.br)

<a id='credits'></a>

## 7. Credits and Acknowledgements

Thanks <a href="https://www.udacity.com" target="_blank">Udacity</a> for letting me use their logo as favicon for this web app.

I want to express my gratitude to Udacity reviewers for their genuine efforts and time. Their invaluable advice and feedback immensely helped me in completing this project.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<a id='license'></a>

## 8. License
This project is under the license [MIT](./LICENSE).
