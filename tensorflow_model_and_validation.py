import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


'''
The deep learning model contains 10 layers that measures linear regression for
finding the critical temperatuire for superconductors given the stats and values.

To analyse the model, we have a deep learning analysis tool to measure how well the data fits.
'''


class dl_model:

    # Initialising X and y, the X and y have to be split by the user
    def __init__(self, X, y):
        self.X = X
        self.y = y


    # Splitting the data
    def train_test_splt(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test



    # Superconductor Deep Learning Model
    def model_itself(X_train, y_train):
        X_train_shape_1 = X_train.shape[1]
        model = Sequential()
        model.add(Dense(128, input_shape=(X_train_shape_1,), activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(2048, activation='relu'))
        model.add(Dense(2048, activation='relu'))
        model.add(Dense(4096, activation='relu'))

        model.add(Dense(1, activation='linear'))

        model.compile(loss='mean_squared_error',
                    optimizer='adam',
                    metrics=['mae', 'mse'])

        print(model.summary())

        model.fit(
            X_train,
            y_train,
            epochs=1500,
            batch_size=256,
            validation_split=0.1
        )


    # Superconductor Deep Learning Model Analysis
    def model_analysis(self):
        y_test_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        print("Training MSE:", round(mean_squared_error(y_train, y_train_pred),4))
        print("Validation MSE:", round(mean_squared_error(y_test, y_test_pred),4))
        print("\nTraining r2:", round(r2_score(y_train, y_train_pred),4))
        print("Validation r2:", round(r2_score(y_test, y_test_pred),4))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('Predicted vs. actual values', fontsize=14, y=1)
        plt.subplots_adjust(top=0.93, wspace=0)
        
        ax1.scatter(y_test, y_test_pred, s=2, alpha=0.7)
        ax1.set_title('Test set')
        ax1.set_xlabel('Actual values')
        ax1.set_ylabel('Predicted values')
        
        ax2.scatter(y_train, y_train_pred, s=2, alpha=0.7)
        ax2.set_title('Train set')
        ax2.set_xlabel('Actual values')
        ax2.set_ylabel('')
        ax2.set_yticklabels(labels='')
        
        plt.show()
