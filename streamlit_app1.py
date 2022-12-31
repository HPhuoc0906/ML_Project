import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

st.header('Welcome !')
upload_file = st.file_uploader('Upload your file here', type= '.csv')
df = None
if upload_file is not None:
    df = pd.read_csv(upload_file)
    if st.button("Show Dataset"):
        st.write(df)
target_prediction = None
target_prediction = st.text_input('Please input your true target prediction:',)    

if (upload_file is not None) and (target_prediction is not None):
    if target_prediction in df.columns:
        Y = np.array(df[target_prediction]).reshape(-1, 1)
        X = df
        X = np.array(X.drop(columns=[target_prediction]))
        st.write('Enter your desire test size :')
        input_test_size = st.text_input('**Note: Test size must be greater than 0 and less or equal 1**')
        try:
            test_size = float(input_test_size)
            if test_size > 0 and test_size <= 1:
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= test_size)
                model = LinearRegression()
                model.fit(X_train, Y_train)
                Y_predict = model.predict(X_test)
                Loss_Value = (Y_predict - Y_test) ** 2
                if st.button('Predict'):
                    newData = {
                        'Data' : X_test.reshape(-1),
                        'Answer' : Y_test.reshape(-1),
                        'Predict' : Y_predict.reshape(-1),
                        'Loss Value' : Loss_Value.reshape(-1)
                    }
                    newDf = pd.DataFrame(newData)
                    st.write(newDf)
                if st.button('Plot Chart'):
                    st.write('Visualize chart of training data')
                    fig, ax = plt.subplots()
                    ax.scatter(X_train, Y_train, c = 'Red')
                    ax.plot(X_train, model.predict(X_train), c = 'Blue')
                    ax.set_xlabel(df.columns[0])
                    ax.set_ylabel(df.columns[1])
                    ax.set_title("Traning Result")
                    st.pyplot(fig)
                    st.write('Visualize chart of test data')
                    fig, ax = plt.subplots()
                    ax.scatter(X_test, Y_test, c = 'r')
                    ax.plot(X_test, Y_predict, c = 'b')
                    ax.set_xlabel(df.columns[0])
                    ax.set_ylabel(target_prediction)
                    ax.set_title("Test Result")
                    st.pyplot(fig)
        except ValueError:
            pass
else: st.write('Please fill out all information !')
