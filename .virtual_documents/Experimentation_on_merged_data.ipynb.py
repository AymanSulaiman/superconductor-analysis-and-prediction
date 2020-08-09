import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('merged.csv')
df


cols = [i for i in df.columns]
# cols


X = df.drop(['critical_temp', 'material'], axis=1)
# X


y = df.critical_temp.values.reshape(-1,1)
# y


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_test, y_train, y_test


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

model_linear = LinearRegression()
model_linear.fit(X_train, y_train)

r2_training = model_linear.score(X_train,y_train)
r2_testing = model_linear.score(X_test, y_test)

print(f'''
Training R^2 Scores: {round(100*r2_training,2)}%
Testing R^2 Scores:  {round(100*r2_testing,2)}%
''')


from xgboost import XGBRegressor
from sklearn.metrics import r2_score

regressor = XGBRegressor()

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
score = r2_score(y_test, y_pred)
score 


# from sklearn.model_selection import GridSearchCV

# xgb1 = XGBRegressor()
# parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
#               'objective':['reg:linear'],
#               'learning_rate': [.03, 0.05, .07], #so called `eta` value
#               'max_depth': [5, 6, 7],
#               'min_child_weight': [4],
#               'silent': [1],
#               'subsample': [0.7],
#               'colsample_bytree': [0.7],
#               'n_estimators': [500]}

# xgb_grid = GridSearchCV(regressor,
#                         parameters,
#                         cv = 2,
#                         n_jobs = 5,
#                         verbose=True)

# xgb_grid.fit(X_train,
#          y_train)
# print('---------------------------------------------------------------------')
# print(xgb_grid.best_score_)
# print(xgb_grid.best_params_)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
epochs = 100
# def make_model(epochs = 2000):
model = Sequential()
model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
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



# model.fit(
#     X_train,
#     y_train,
#     epochs=epochs,
#     batch_size=512,
#     validation_split=0.1,
# )
    
#     return model


history = model.fit(X_train, y_train, epochs=1, batch_size=512, validation_data=(X_test, y_test))


def deep_learning_model_evaluation(model, skip_epochs=0, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):
    
    from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score

    # MSE and r squared values
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    print("Training MSE:", round(mean_squared_error(y_train, y_train_pred),4))
    print("Validation MSE:", round(mean_squared_error(y_test, y_test_pred),4))
    print("\nTraining r2:", round(r2_score(y_train, y_train_pred),4))
    print("Validation r2:", round(r2_score(y_test, y_test_pred),4))
    

    # Scatterplot of predicted vs. actual values
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


history = model.fit(X_train, y_train, epochs=1000, batch_size=512, validation_data=(X_test, y_test))
pd.DataFrame(history.history).plot(figsize=(15,7))
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('metric value')
# plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()
deep_learning_model_evaluation(model=model)


from tensorflow import keras


def build_model(n_hidden=10, n_neurons=300, learning_rate=3e-3, input_shape=(X_train.shape[1],)):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model


keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)


keras_reg.fit(X_train, y_train, epochs=100,
              validation_data=(X_test, y_test),
              callbacks=[keras.callbacks.EarlyStopping(patience=10)]
             )


mse_test = keras_reg.score(X_test, y_test)


from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(1, 100),
#     "learning_rate": reciprocal(3e-4, 3e-2),
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3, verbose=2)
rnd_search_cv.fit(X_train, y_train, epochs=100,
                  validation_data=(X_test, y_test),
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])


rnd_search_cv.best_params_


rnd_search_cv.best_score_


rnd_search_cv.best_estimator_


rnd_search_cv.score(X_test, y_test)


model = rnd_search_cv.best_estimator_.model
model


model.evaluate(X_test, y_test)






