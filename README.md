# 🏠 USA Housing Price Prediction: A Regression-Based Approach

Welcome to my data science project! This repository showcases a complete machine learning workflow to predict housing prices in the USA using various regression techniques.

---

## 🎯 Goal of the Project

Housing prices are influenced by a wide range of features such as income levels, number of rooms, age of the house, and more. The goal of this project is to:
- Understand the relationship between these variables and housing prices.
- Build and evaluate predictive models using *linear regression techniques*.
- Interpret the model results to gain actionable insights.

---

## 📁 Dataset Overview

This project uses a *Kaggle-sourced dataset* containing several variables that influence house pricing. Key features include:

- Avg_Area_Income: Average income in the area
- Avg_Area_House_Age: Age of houses in the area
- Avg_Area_Number_of_Rooms: Average number of rooms
- Avg_Area_Number_of_Bedrooms: Average number of bedrooms
- Area_Population: Population of the area
- Price: Target variable (house price)
- Address: Categorical feature used for location encoding

---

## 🛠 Tools and Technologies

The project is built using the following tools:

- *Python* – Programming
- *Pandas & NumPy* – Data handling and numerical processing
- *Matplotlib & Seaborn* – Visualization
- *Scikit-learn* – Model building and evaluation
- *Statsmodels* – Statistical summary and diagnostics

---

## 📊 Data Analysis and Preprocessing

We began with data cleaning and exploration:
- Checked for missing values
- Explored distributions using histograms and boxplots
- Visualized correlation between variables using a heatmap
- Detected and treated outliers using *quantile-based capping*
- Encoded the Address feature using *label encoding*

---

## 🧠 Machine Learning Models Implemented

A variety of regression models were trained and compared:

| Model               | Description                                      |
|--------------------|--------------------------------------------------|
| OLS Regression      | Provides statistical overview of predictors     |
| Linear Regression   | Baseline regression model                       |
| Ridge Regression    | Penalizes large coefficients to reduce overfitting |
| Lasso Regression    | Performs feature selection by shrinking weights |
| SGD Regressor       | Uses gradient descent for scalable training     |

---

## 📐 Model Evaluation Metrics

Each model was evaluated using multiple performance metrics:

- *R² Score*: Measures how well the model explains variance
- *Mean Squared Error (MSE)*
- *Mean Absolute Error (MAE)*
- *Variance Inflation Factor (VIF)*: Checked for multicollinearity

---

## 📈 Key Findings

- *Avg_Area_Income* and *Area_Population* were the most influential features.
- *Ridge and Lasso* slightly outperformed the standard linear model in terms of generalization.
- *OLS summary* provided statistical insights, including p-values and confidence intervals.
- Visual inspection of residual plots showed good model fit with minimal heteroscedasticity.

---

## 💡 What I Learned

- Built an understanding of *how and when to use Ridge vs Lasso vs OLS*.
- Learned how to use *VIF* to diagnose multicollinearity.
- Understood the importance of *data preprocessing and encoding*.
- Gained confidence in *model evaluation and interpretation techniques*.

---
