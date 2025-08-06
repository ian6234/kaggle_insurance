import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_log_error, make_scorer, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures

from sklearn.impute import SimpleImputer

import scipy

import time
import datetime

import pickle


def data_cleaning(file):

    train = pd.read_csv(f'{file}.csv')
    # ---------------------Data Cleaning-------------------------------------------

    # Replace Missing Numerical Values with mean
    mean_imputer = SimpleImputer(strategy='mean')

    train['Age'] = mean_imputer.fit_transform(train[['Age']])
    train['Annual Income'] = mean_imputer.fit_transform(train[['Annual Income']])
    train['Number of Dependents'] = mean_imputer.fit_transform(train[['Number of Dependents']])
    train['Health Score'] = mean_imputer.fit_transform(train[['Health Score']])
    train['Previous Claims'] = mean_imputer.fit_transform(train[['Previous Claims']])
    train['Vehicle Age'] = mean_imputer.fit_transform(train[['Vehicle Age']])
    train['Credit Score'] = mean_imputer.fit_transform(train[['Credit Score']])
    train['Insurance Duration'] = mean_imputer.fit_transform(train[['Insurance Duration']])

    # Replace Missing Categorical data with mode
    # category_imputer = SimpleImputer(strategy='most_frequent')

    # train['Education Level'] = category_imputer.fit_transform(train[['Education Level']]).ravel()
    # train['Exercise Frequency'] = category_imputer.fit_transform(train[['Exercise Frequency']]).ravel()

    # Replace Missing Categorical data with 'Missing'
    train['Gender'] = train['Gender'].fillna('Missing')
    train['Marital Status'] = train['Marital Status'].fillna('Missing')
    train['Occupation'] = train['Occupation'].fillna('Missing')
    train['Location'] = train['Location'].fillna('Missing')
    train['Policy Type'] = train['Policy Type'].fillna('Missing')
    train['Customer Feedback'] = train['Customer Feedback'].fillna('Missing')
    train['Smoking Status'] = train['Smoking Status'].fillna('Missing')
    train['Property Type'] = train['Property Type'].fillna('Missing')
    train['Education Level'] = train['Education Level'].fillna('Missing')
    train['Exercise Frequency'] = train['Exercise Frequency'].fillna('Missing')

    train.to_csv(f'{file}_cleaned.csv')
    print('data cleaned!')


def data_encoding(file):
    train = pd.read_csv(f'{file}_cleaned.csv')

    # ---------------------Data Encoding-------------------------------------------

    # Ordinal Encoding
    # education = {'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3}
    # train['Education Level'] = train['Education Level'].map(education)

    # Target Encoding
    # exercise = {'Daily': 30, "Weekly": 4, "Monthly": 1, 'Rarely': 0, 'Missing': 0}
    # train['Exercise Frequency'] = train['Exercise Frequency'].map(exercise)

    time_format = '%Y-%m-%d %H:%M:%S.%f'
    train['Policy Start Date'] = train['Policy Start Date'].apply(
        lambda x: (time.time() - time.mktime(datetime.datetime.strptime(x, time_format).timetuple())) / (3600*24*365))

    train['Policy Start Date'] = train['Policy Start Date'].fillna(0)

    # One Hot Encoding
    train = pd.get_dummies(train, columns=['Gender', 'Smoking Status', 'Marital Status', 'Occupation', 'Location',
                                           'Policy Type', 'Customer Feedback', 'Property Type', 'Education Level', 'Exercise Frequency'], drop_first=True)

    train = train.replace({True: 1, False: 0})

    train.to_csv(f'{file}_encoded.csv', index=False)

    print('data encoded!')


# Define the RMSLE function
def log_error(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))


target_features = ['Previous Claims', 'Annual Income', 'Health Score', 'Credit Score', 'Smoking Status_Yes',
                   'Occupation_Unemployed']


def model_testing():
    data = pd.read_csv('train_encoded.csv')

    # log_cols = ['Annual Income', 'Health Score']
    # data[log_cols] = np.log1p(data[log_cols])

    x = data.drop(columns=['Premium Amount'])

    y = data['Premium Amount']
    y = np.log1p(y)

    # poly = PolynomialFeatures(degree=2)  # Change the degree for higher-degree polynomials
    # x = poly.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1,  max_depth=3)

    model = LinearRegression()
    model.fit(x_train, y_train)

    #importances = model.feature_importances_
    #sorted_idx = importances.argsort()
    #feature_names = x.columns
    # Plot
    #plt.barh(feature_names[sorted_idx], importances[sorted_idx])
    #plt.xlabel("Feature Importance")
    #plt.show()

    pred = model.predict(x_test)

    r2 = r2_score(y_test, pred)
    print(f'r-squared test score (log premium): {r2}')

    pred = np.expm1(pred)
    y_test = np.expm1(y_test)

    train_pred = model.predict(x_train)

    rmsle = root_mean_squared_log_error(y_test, pred)
    print(f'root mean squared log error: {rmsle}')

    r2_train = r2_score(y_train, train_pred)
    print(f'r-squared training score: {r2_train}')

    counts, bins = np.histogram(pred, bins=200)
    counts_2, bins_2 = np.histogram(y_test, bins=200)
    plt.stairs(counts, bins, color='blue')
    plt.stairs(counts_2, bins_2, color='red')
    plt.show()


def run_submission():
    train = pd.read_csv('train_encoded.csv')

    x_test = pd.read_csv('test_encoded.csv')

    ids = x_test['id']

    x_train = train.drop(columns=['Premium Amount'])
    # x_train = train[target_features]

    # x_test = x_test[target_features]

    y_train = train['Premium Amount']
    y_train = np.log1p(y_train)

    # model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=6, subsample=0.9)
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,  max_depth=4)

    # model.n_jobs = 6

    model.fit(x_train, y_train)

    prediction = model.predict(x_test)

    prediction = np.expm1(prediction)

    x_test['Premium Amount'] = prediction
    x_test['id'] = ids

    submission = x_test[['id', 'Premium Amount']]
    submission.to_csv('submission.csv', index=False)


def show_distributions():
    df = pd.read_csv('train_encoded.csv')
    df = np.log1p(df)
    # Set up grid size based on the number of features
    num_features = df.shape[1]
    cols = 3  # Number of columns in the grid
    rows = (num_features + cols - 1) // cols  # Calculate rows needed

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten()  # Flatten in case of a 2D array of axes

    # Plot each feature
    for i, col in enumerate(df.columns):
        ax = axes[i]
        if np.issubdtype(df[col].dtype, np.number):  # Numeric data
            ax.hist(df[col].dropna(), bins=50, color='blue', alpha=0.7)
        else:  # Categorical data
            df[col].value_counts().plot(kind='bar', ax=ax, color='orange', alpha=0.7)
        ax.set_title(col)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

    # Hide unused subplots if any
    for i in range(num_features, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('feature_histograms_log.png', dpi=300)
    plt.show()


model_testing()






