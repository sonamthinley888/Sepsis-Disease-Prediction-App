import streamlit as st
import sklearn
from sklearn.utils import shuffle
from sklearn import datasets
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

# Load libraries
import numpy
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# Data file import
sepsis_data = pd.read_csv("sepsis_file.csv")
df = sepsis_data
# Attribute to be predicted
predict = "Sepsis"
st.title('Predictive Data Analysis')
#pre-processing
st.subheader('Pre Processing Data')
# Removing Patient ID and Insurance
df = df[["PRG", "PL", "PR", "SK", "TS", "M11", "BD2", "Age", "Sepsis"]]

from sklearn.exceptions import DataDimensionalityWarning
#encode object columns to integers
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder

for col in df:
  if df[col].dtype =='object':
    df[col]=OrdinalEncoder().fit_transform(df[col].values.reshape(-1,1))
df

st.subheader('Normalizing the independent values of the dataframe')
class_label =df['Sepsis']
df = df.drop(['Sepsis'], axis =1)
df = (df-df.min())/(df.max()-df.min())
df['Sepsis']=class_label
df


#pre-processing
sepsis_data = df.copy()
le = preprocessing.LabelEncoder()

PRG = le.fit_transform(list(sepsis_data["PRG"])) # Plasma glucose
PL = le.fit_transform(list(sepsis_data["PL"])) # Blood Work Result-1 (mu U/ml)
PR = le.fit_transform(list(sepsis_data["PR"])) # Blood Pressure (mm Hg)
SK = le.fit_transform(list(sepsis_data["SK"])) # Blood Work Result-2 (mm)
TS = le.fit_transform(list(sepsis_data["TS"])) # Blood Work Result-3 (mu U/ml)
M11 = le.fit_transform(list(sepsis_data["M11"])) # Body mass index (weight in kg/(height in m)^2
BD2 = le.fit_transform(list(sepsis_data["BD2"])) # Blood Work Result-4 (mu U/ml)
Age = le.fit_transform(list(sepsis_data["Age"])) # Patients Age

Sepsis = le.fit_transform(list(sepsis_data["Sepsis"])) # Sepsis 0-not present 1-present


x = list(zip(PRG, PL, PR, SK, TS, M11, BD2, Age))
y = list(Sepsis)
# Test options and evaluation metric
num_folds = 5
seed = 7
scoring = 'accuracy'

# Model Test/Train
# Splitting what we are trying to predict into 4 different arrays -
# X train is a section of the x array(attributes) and vise versa for Y(features)
# The test data will test the accuracy of the model created
import sklearn.model_selection
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.20, random_state=seed)
#splitting 20% of our data into test samples. If we train the model with higher data it already has seen that information and knows

#size of train and test subsets after splitting
st.subheader('Size of train and test subsets after splitting')
np.shape(x_train), np.shape(x_test)

models = []
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))



# evaluate each model in turn
results = []
names = []
st.subheader("Performance on Training set")
for name, model in models:
  kfold = KFold(n_splits=num_folds,shuffle=True,random_state=seed)
  cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  msg += '\n'
  st.write(msg)

st.subheader('Algorithms Performance')
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
st.pyplot(fig)

st.subheader('Evaluation by testing with independent/external test data set')
st.write("Make predictions on validation/test dataset")

models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
dt = DecisionTreeClassifier()
nb = GaussianNB()
gb = GradientBoostingClassifier()
rf = RandomForestClassifier()

best_model = rf
best_model.fit(x_train, y_train)
y_pred = best_model.predict(x_test)
st.write("Best Model Accuracy Score on Test Set:", accuracy_score(y_test, y_pred))

st.subheader('Model Performance Evaluation Metric 1 - Classification Report')
report = classification_report(y_test, y_pred)
st.text(report)

st.subheader('Model Performance Evaluation Metric 2 - Confusion matrix')
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
st.pyplot()


st.subheader('Model Evaluation Metric 4- Prediction Report')
for x in range(len(y_pred)):
  st.write("Predicted: ", y_pred[x], "Actual: ", y_test[x], "Data: ", x_test[x],)