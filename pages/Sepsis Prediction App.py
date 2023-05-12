import streamlit as st
from model import *
import numpy as np

# creating a function for Prediction
def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = best_model.predict(input_data_reshaped)

    print(prediction)

    accuracy = str(round(model_accuracy,3))

    if (prediction[0] == 0):
        return 'The patient has a less chance of getting Sepsis. Accuracy: ' + accuracy

    else:
        return 'The patient has a high chance of getting Sepsis. Accuracy: ' + accuracy





def main():
    # giving a title
    st.title('Sepsis Prediction App')

    # getting the input data from the user
    PRG = st.text_input('Plasma Glucose')
    PL = st.text_input('Blood Work Result-1 (mu U/ml)')
    PR = st.text_input('Blood Pressure (mm Hg)')
    SK = st.text_input('Blood Work Result-2 (mm)')
    TS = st.text_input('Blood Work Result-3 (mu U/ml)')
    M11 = st.text_input('Body mass index (weight in kg/(height in m)^2')
    BD2 = st.text_input('Blood Work Result-4 (mu U/ml)')
    Age = st.text_input('Patients Age (years)')


    # code for Prediction
    diagnosis = ''


    # creating a button for Prediction

    if st.button('Sepsis Test Result'):
        diagnosis = diabetes_prediction(
            [PRG,PL,PR,SK,TS,M11,BD2, Age])

    st.success(diagnosis)


if __name__ == '__main__':
    main()
