# What Makes a High-rating Movie: Reviews Mining and Rating Prediction

## 1. Project Description

In this project, we trying to use Machine Learning, Deep Learning and NLP models to understand and predict the movie rating in IMDB. The final selected model is Light GBM model with inputs of both basic features and text features. And the results shows that review sentiment score, revenue, budget and runtime are the most important features that impact movie rating.

## 2. EDA & Data Preprocessing

**All data used in this project are from *Kaggle.com*:**

IMDB Dataset: 

https://www.kaggle.com/ashirwadsangwan/imdb-dataset

The movies Dataset: 

https://www.kaggle.com/rounakbanik/the-movies-dataset

IMDB Review Dataset: 

https://www.kaggle.com/ebiswas/imdb-review-dataset

### 2.1 Exploratory Data Analysis

*EDA.ipynb*

### 2.2 Data Pre-processing

*Data_preprocess_MLP.ipynb*

## 3. Movie Overviews and Reviews Analysis

In this part, we try to extract information from the overview and review data.

### 3.1 Overviews and Keywords Analysis

#### 3.1.1 Overviews Analysis

*Movie_overview_DistilBert.ipynb*

#### 3.1.2 Keywords Analysis

*BOW+Onehotencoder+TFIDF+PCA.ipynb*

### 3.2 Reviews Analysis

*movie_review.ipynb*

## 4. Rating Prediction

In this part, we are trying to use machine learning and deep learning models to predict the average rating of each movie using both basic metadata and text features generated from NLP models.

We try both Regression and Classification models to find more insights. For Regression models, the target variable is average rating ranged from 0-10. For Classification models, we set target equals to 1 when average rating greater than or equal to 6.5, and 0 otherwise.

Also, for the input features, we try both basic features and basic features plus text features.

### 4.1 Linear Regression & Logistic Regression - Baseline Models

*Data_preprocess_MLP.ipynb*

### 4.2 Tree Models

#### Regression:

**With basic features only:** 

models_regression_without_text_tuning.ipynb

**With basic, overview and review features:**

models_regression_with_text_tuning_xgboost+linear model.ipynb

models_regression_with_text_tuning_decision tree.ipynb

models_regression_with_text_tuning_lightgbm.ipynb

#### Classification

**With basic features only:** 

*classification_without_text.ipynb*

**With basic, overview and review features:**

*classification_with_text.ipynb*

### 4.3 Multilayer Perceptron (MLP)

*Data_preprocess_MLP.ipynb*

### 4.4 Summary

The regression model based on LightGBM performs the best, with an MSE of **0.62** on the test set. It uses basic data, features extracted from the overview and the review to train.

The distribution of high rating movie between the train and test dataset are shown in the figures below. We can see the distribution are balanced. We chose accuracy as evaluation metrics, since we care about the classification of both classes. The highest accuracy is **76.76%**.

## 5. Explainable AI & Insights

In this part, the model based on LightGBM with 865 features is further explore.

For the regression model:

*data_preprocess_for_lime.ipynb*

*final_regression_model_interpretation.ipynb*

For the classification model:

*classification_with_text.ipynb*

## 6. Image Classification

In the IMDB dataset, there is a feature called *poster_path*, which contains the URLs for poster image of each film. We download poster images from https://image.tmdb.org/, and gather total 5713 poster images from valid URLs.

### 6.1 Exploratory Data Analysis

*Image_EDA.ipynb*

### 6.2 Image Classification Model and Analysis

In order to balance two classes and save ram, we extract a sub-dataset with 2000 images, 1000 images with positive targets and 1000 with negative targets. 

For the target variable, we also treat them as previous classification models that target equals to 1 when average rating greater than or equal to 6.5, and 0 otherwise. For the image matrix, we also normalize pixel number to between 0 and 1 by dividing 255.

*Image_classification.ipynb*

## 7. Conclusion

In this project, we trying to use Machine Learning, Deep Learning and NLP models to understand and predict the movie rating in IMDB. Here are the conclusions that we gain through this project.

(1) For overview feature extraction, features generated from BERT can achieve most accurate results.

(2) In review analysis part, the scores of the sentiment analysis are concentrated at 0.6, so most of the viewers in the sample have a positive evaluation of the movie. The most important concern in positive evaluation is actors' acting performance and storyline.

(3) In rating prediction part, among both regression and classification models, Light GBM can achieve smallest MSE and highest accuracy. Our final prediction model is Light GBM model with inputs of both basic features and text features.

(4) In interpretation part, review sentiment score, revenue, budget and runtime are the most important features that impact average rating.

(5) In the trial of image classification, the results shows that movie poster may has little impact on viewers’ rating.

According to the conclusions, we have some strategy suggestions for movie companies that may help to gain a higher rating among viewers.

(1) Before movie comes out, improve movie budget and runtime can help to get higher rating.

(2) After movie comes out, higher review sentiment score and higher revenue can lead to higher rating. Besides, review sentiment score is highly related to actors' acting performance and storyline.

