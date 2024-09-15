# Titanic Survival Prediction

## Table of Contents
- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset) / [Installation](#installation)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Contributors](#contributors)

## Project Overview
The **Titanic Survival Prediction** project uses machine learning to predict whether a passenger survived or not, based on the attributes provided in the dataset, such as age, gender, class, and more. This project applies **Logistic Regression**, a popular classification algorithm, to classify passengers into those who survived and those who didn’t.

The objective is to accurately predict survival rates by learning from the given training data and evaluating the model’s performance on a test dataset.

## Tech Stack
- **Programming Language**: Python
- **Libraries**:
  - `pandas` - For data manipulation and analysis
  - `numpy` - For numerical operations
  - `scikit-learn` - For building and evaluating machine learning models
  - `matplotlib` and `seaborn` - For data visualization

## Dataset and Installation
The Titanic dataset consists of information about the passengers aboard the Titanic. The key features in the dataset include:
- **PassengerId**: Unique ID for each passenger
- **Pclass**: Ticket class (1st, 2nd, or 3rd class)
- **Name**: Name of the passenger
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger
- **SibSp**: Number of siblings or spouses aboard
- **Parch**: Number of parents or children aboard
- **Ticket**: Ticket number
- **Fare**: Ticket fare paid
- **Cabin**: Cabin number
- **Embarked**: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
- **Survived**: Survival status (0 = No, 1 = Yes) (Target variable)

You can download the dataset from [Kaggle's Titanic Data](https://www.kaggle.com/c/titanic/data).


## Model Training and Evaluation
1. **Data Preprocessing**:
   - Handled missing values in columns such as `Age` and `Embarked` using mean/most frequent value imputation.
   - Converted categorical variables like `Sex` and `Embarked` into numerical values using **One-Hot Encoding**.
   - Scaled numerical features such as `Age` and `Fare` using `StandardScaler`.

2. **Model Selection**:
   - Applied **Logistic Regression**, a binary classification algorithm, to predict survival.
   - Split the dataset into **training** and **testing** sets using `train_test_split` from `scikit-learn`.

3. **Model Evaluation**:
   - Evaluated the model using key metrics such as **Accuracy**, **Precision**, **Recall**, and the **Confusion Matrix**.
   - Cross-validation was performed to ensure model robustness and avoid overfitting.

## Results
- The **accuracy** of the Logistic Regression model is approximately **82.1%**.
- Other key metrics include:
  - **Precision**: 78.4%
  - **Recall**: 73.9%
  - **F1-Score**: 76.1%

All evaluation metrics, along with the **Confusion Matrix**, can be viewed in the Jupyter notebook.

## Contributors
- **Kolli Venkata Rushita** - [GitHub](https://github.com/Kolli-VenkataRushita)

Feel free to explore the project and contribute with any suggestions or improvements!
```
