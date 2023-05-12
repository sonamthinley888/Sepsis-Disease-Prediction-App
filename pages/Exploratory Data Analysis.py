import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Web App Title
st.title('Exploratory Data Analysis')

st.markdown('The exploratory data analysis will seek to answer the following questions: \n'
            '1.	What is the number of attributes and observations in the dataset?\n'
            '2.	Are there any missing values in the dataset?\n'
            '3.	How many of the patients are diagnosed with Sepsis?\n'
            '4.	What is the correlation between the variables?\n'
            '5.	What is the distribution of the variables?')

st.subheader('Data Set')
# Data file import
sepsis_data = pd.read_csv("sepsis_file.csv")
df = sepsis_data
st.write(df)
st.subheader('Data Fields')
st.markdown('''

        Name             | Attribute/Target | Description

        | ID             | N/A              | Unique number to represent patient ID

        | PRG            | Attribute1       |  Plasma glucose| 

        | PL             | Attribute 2      |   Blood Work Result-1 (mu U/ml)

        | PR             | Attribute 3      | Blood Pressure (mm Hg)|

        | SK             | Attribute 4      | Blood Work Result-2 (mm)|

        | TS              | Attribute 5      |     Blood Work Result-3 (mu U/ml)|

        | M11             | Attribute 6      |  Body mass index (weight in kg/(height in m)^2|

        | BD2             | Attribute 7      |   Blood Work Result-4 (mu U/ml)|

        | Age             | Attribute 8      |    patients age  (years)|

        | Insurance      | N/A              | If a patient holds a valid insurance card|

        | Sepsis        | Target ''')

# Removing Patient ID and Insurance
st.subheader('Removing Patient ID and Insurance from Dataset')
df = df[["PRG", "PL", "PR", "SK", "TS", "M11", "BD2", "Age", "Sepsis"]]
st.write(df)

st.subheader('Converting Target Variable to integer')
from sklearn.preprocessing import OrdinalEncoder

for col in df:
    if df[col].dtype == 'object':
        df[col] = OrdinalEncoder().fit_transform(df[col].values.reshape(-1, 1))
df

st.header('Data Statistics')
st.write(df.describe())

st.header('Data Head')
st.write(df.head())

st.header('Data Tail')
st.write(df.tail())

st.header('EDA Questions ')
st.subheader('Q1. What is the number of attributes and observations in the dataset?')
st.text('rows and columns-data shape(attributes & samples)')
df.shape

# unique values for each attribute
st.text('unique values for each attribute')
st.write(df.nunique())

st.subheader('Q2. Are there any missing values in the dataset?')
# Find the missing values in the rows
df[df.isnull().any(axis=1)]
# drop rows with any missing values
df = df.dropna()
st.text('Dropping the rows with missing values')

st.subheader('Q3. How many of the patients are diagnosed with Sepsis?')
# checking target value distribution
print(df.Sepsis.value_counts())
fig, ax = plt.subplots(figsize=(5, 4))
name = ["Negative", "Positive"]
ax = df.Sepsis.value_counts().plot(kind='bar')
ax.set_title("Sepsis Classes", fontsize=13, weight='bold')
ax.set_xticks([0, 1])
ax.set_xticklabels(name, rotation=0)

# To calculate the percentage
totals = []
for i in ax.patches:
    totals.append(i.get_height())
total = sum(totals)
for i in ax.patches:
    ax.text(i.get_x() + .09, i.get_height() - 50, \
            str(round((i.get_height() / total) * 100, 2)) + '%', fontsize=14,
            color='white', weight='bold')

plt.tight_layout()
st.pyplot(fig)

st.subheader('Q4. What is the correlation between the variables?')
# check correlation between variables
sns.set(style="white")
plt.rcParams['figure.figsize'] = (15, 10)
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, linewidths=.5, cmap="Blues", ax=ax)
plt.title('Corelation Between Variables', fontsize=30)
plt.show()
st.pyplot(fig)
st.text('Blood work result 1 has the highest correlation with target variable Sepsis')
st.text('There is a correlation between Age and Plasma glucose')

# Visualising data  distribution in detail
st.subheader('Q5. What is the distribution of the variables?')
fig = plt.figure(figsize=(18, 18))
ax = fig.gca()
df.hist(ax=ax, bins=30)
plt.show()
st.pyplot(fig)

