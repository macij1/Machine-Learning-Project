# -*- coding: utf-8 -*-
"""ML_120.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HvI3kh94zOCuG4mWoSZpgMn-j2YdNNMo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, KFold
from sklearn import svm, tree, neighbors, neural_network, ensemble, preprocessing
from sklearn.metrics import confusion_matrix, fowlkes_mallows_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import mutual_info_score, adjusted_rand_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.cluster import KMeans

# Binary encoding
def binary_encode_column(df, column_name):
    # Get unique values in the column and the number of bits required
    unique_values = df[column_name].unique()
    num_bits = int(np.ceil(np.log2(len(unique_values))))  # Minimum bits needed

    # Map each unique value to its binary representation
    binary_mapping = {val: format(i, f'0{num_bits}b') for i, val in enumerate(unique_values)}

    # Apply the mapping to create a new column with binary strings
    df[f"{column_name}_bin"] = df[column_name].map(binary_mapping)

    # Drop the original column if desired
    df = df.drop(columns=[column_name])

    return binary_mapping

def create_difficulty(averageGrade, SD):
  difficulty = 0
  if averageGrade > 0 and averageGrade < 2:
    difficulty += 4
  elif averageGrade > 2 and averageGrade < 2.7:
    difficulty += 3
  elif averageGrade > 2.7 and averageGrade < 3.3:
    difficulty += 2
  elif averageGrade > 3.3 and averageGrade < 3.7:
    difficulty += 1
  if SD > 0.5:
    difficulty += 1
  if difficulty > 4:
    difficulty = 4
  return difficulty

def preprocess_data(filename = 'Grade_Distribution_Data.xlsx'):
    # Get data and show that it has done so, Note you will have to upload the data if opened on google Colab, file in teams chat
    data = pd.read_excel(filename, sheet_name='AY2023 AY2024 Grade Distro')
    data.head()
    
    # Begin preprocessing here
    # There is a lot more that can be done with pandas for manipulating our data, this is just a start
    train_data = data.copy(deep="True")
    np.random.seed(100)

    train_data.drop(columns=['A', 'B', 'C', 'D', 'F', 'W', 'Trm Code'], inplace=True)
    train_data.rename(columns={'Academic Year': 'AcademicYear', 'Course Subject and Number': 'CourseSubjectandNumber', 'Average Grade': 'AverageGrade', 'Primary Instructor Name': 'PrimaryInstructorName'}, inplace=True)
    train_data = train_data[train_data.AcademicYear != "2022-23"]
    train_data = train_data[train_data.AverageGrade != 'Total']
    train_data.drop(columns=['AcademicYear'], inplace=True)
    train_data.dropna(inplace=True)

    #remove any labs, recitations, etc
    train_data = train_data[~train_data['Section'].str.contains(r'[0-9]', regex=True, na=False)]
    train_data.drop(columns=['Section'], inplace=True)

    #Begin converting all strings into ints, this require a lot of encoding and will be difficult for instructors
    train_data[['Subject', 'Number']] = train_data['CourseSubjectandNumber'].str.split(' ', n=1, expand=True)
    train_data.drop(columns=['CourseSubjectandNumber'], inplace=True)
    train_data.rename(columns={'PrimaryInstructorName': 'Instructor'}, inplace=True)
    #print(train_data.columns)

    columns_map = {'Subject': {'AE': 0, 'ARCH': 1, 'ECE': 2, 'ME': 3, 'NRE': 4, 'AE': 5, 'MP': 6}}
    train_data.replace(columns_map, inplace=True)

    #Apply difficulty to each column
    train_data.insert(len(train_data.columns), "Difficulty", 0)
    train_data['Difficulty'] = train_data.apply(lambda row: create_difficulty(row['AverageGrade'], row['Standard Deviation']),axis=1)

    # TODO: Recover name with the mapping
    instructor_encode_map = binary_encode_column(train_data, 'Instructor')
    train_data.drop(columns=['Instructor'], inplace=True)


    #Easy way to split into train and test, only train is used right now
    train, test = np.split(train_data.sample(frac=1), [int(0.8*len(train_data))])

    X_train = train.loc[:, train_data.columns != 'Difficulty']
    y_train = train.loc[:, 'Difficulty']
    return X_train, y_train

#KMeans
def kmeans(X_train, y_train):
    model = KMeans(n_clusters=5, random_state=0, n_init="auto") #Basic, no parameter tweaking really
    km_model = model.fit(X_train.values) # Need everything to be numbers instead of strings

    # confusion matrix
    KM_y_pred = cross_val_predict(model, X_train, y_train) #Default parameters once again, perhaps specify number of folds?

    KM_tn, KM_fp, KM_fn, KM_tp = confusion_matrix(y_train, KM_y_pred).ravel()
    # print("True Negatives", KM_tn)
    # print("False Positives", KM_fp)
    # print("False Negatives", KM_fn)
    # print("True Positives", KM_tp)

    # silhouette = silhouette_score(X_train, km_model.labels_)
    # db_index = davies_bouldin_score(X_train, km_model.labels_)
    # ch_index = calinski_harabasz_score(X_train, km_model.labels_)
    # print(f"Silhouette Score: {silhouette:.2f}")
    # print(f"Davies-Bouldin Index: {db_index:.2f}")
    # print(f"Calinski-Harabasz Index: {ch_index:.2f}")

    # fm_score = fowlkes_mallows_score(y_train, km_model.labels_)
    # print(f"Fowlkes-Mallows Score: {fm_score:.2f}")
    return KM_y_pred

# 2D Kmeans to try visualization. Irrelevant values
def two_d_Kmeans(X_train, y_train):
    Xv = X_train[['AverageGrade', 'Number']]

    # Standard scaling
    scaler = preprocessing.StandardScaler()
    scaled = scaler.fit_transform(Xv)
    Xv_scaled = pd.DataFrame(scaled, columns=Xv.columns)

    model = KMeans(n_clusters=4, random_state=0, n_init="auto") #Basic, no parameter tweaking really
    labels = model.fit_predict(Xv_scaled, y_train) # Need everything to be numbers instead of strings

    return Xv, Xv_scaled, labels

