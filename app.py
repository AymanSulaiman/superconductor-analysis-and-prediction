import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

st.title('Superconductor Machine Learning App And Analysis')




path_merged = os.path.join('data','merged.csv')
path_train = os.path.join('data','train.csv')
path_unique_m = os.path.join('data','unique_m.csv')
path_data_no_elements = os.path.join('data','data_no_elements.csv')


df_merged = pd.read_csv(path_merged)
df_train = pd.read_csv(path_train)
df_unique_m = pd.read_csv(path_unique_m)
df_data_no_elements = pd.read_csv(path_data_no_elements)


X = df_data_no_elements.drop(['critical_temp'], axis=1)
y = df_data_no_elements['critical_temp'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_no_material, X_test_no_material = X_train.drop(['material'], axis=1), X_test.drop(['material'], axis=1)



regressor = XGBRegressor()
regressor.load_model('model_from_data_no_elements.txt')


st.sidebar.header('User Input Features')

st.sidebar.selectbox('Select a material',([i for i in X_train.material]))





######################################
# Showing the data and visualizations#
######################################

st.write('''
This is an app that predicts the Critical Temperature of a Superconductor 
using XGBoost.  Shoutout to Kam Ham idieh of UPenn for donating the data to UC Irvine for providing the clean data 
so I can make this app. Here is the link to the [dataset]('https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data')
and the [Data Professor]('https://www.youtube.com/channel/UCV8e2g4IWQqK71bbzGDEI4Q') who's tutorial I followed for the streamlit app.  
''')

show_training_set_df = df_data_no_elements.tail(n=round(0.2*21263))
show_training_set_df

st.title(f'Number of rows: {len(df_merged)}')


def mean_atomic_mass_and_critical_temperature():
    fig = px.scatter(
        df_merged,
        hover_data=['material'],
        x='mean_atomic_mass',
        y='critical_temp',
        size='critical_temp', 
        color='number_of_elements',
        width=1000, 
        # height=800
    )


    fig.update_layout(
        title='Mean Atomic Mass and Critical Temperature',
        xaxis=dict(
            title='Mean Atomic Mass'
        ),
        yaxis=dict(
            title='Critical Temperature (K)'
        ),
        # margin=dict(l=50, r=50, t=100, b=100),
    )

    return st.plotly_chart(fig)

mean_atomic_mass_and_critical_temperature()

st.write('''
## The Machine Learning Algorithm
''')

X = df_merged.drop(['critical_temp', 'material'], axis=1)
y = df_merged['critical_temp'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = XGBRegressor()
regressor.load_model('model.txt')

y_test_pred = regressor.predict(X_test)
score_test = r2_score(y_test, y_test_pred)

st.write(f'''
Score Test: {100*round(score_test,4)} %
''')

st.write(f'''
[GitHub Link](#)    
[Resume]()      
[LinkedIn]()
''')