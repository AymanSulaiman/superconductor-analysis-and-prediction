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

st.title('Superconductor Analysis and Machine Learning App')

st.write('''
This is an app that predicts the Critical Temperature of a Superconductor 
using XGBoost.  Shoutout to UC Irvine for providing the clean data 
so I can make this app. Here is the link to the [dataset]('https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data')
and the [Data Professor]('') who's tutorial I followed for the streamlit app.  
''')

st.sidebar.header('User Input Features')


path_merged = os.path.join('data','merged.csv')
path_train = os.path.join('data','train.csv')
path_unique_m = os.path.join('data','unique_m.csv')
path_data_no_elements = os.path.join('data','data_no_elements')


df_merged = pd.read_csv(path_merged)
df_train = pd.read_csv(path_train)
df_unique_m = pd.read_csv(path_unique_m)
df_data_no_elements = pd.read_csv(path_data_no_elements)


X = df_merged.drop(['critical_temp', 'material'], axis=1)
y = df_merged['critical_temp'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = XGBRegressor()
regressor.load_model('model.txt')


# @st.cache(suppress_st_warning=True)
# def show_dataframe(x):
#     if x == 'm':
#         return df_merged
#     elif x == 't':
#         return df_train
#     elif x == 'u':
#         return df_unique_m
#     else:
#         return 'you done goofed'

# st.title('train')
# st.write(show_dataframe(x = 't'))
# st.table(df_train)

# st.title('unique')
# st.write(show_dataframe(x = 'u'))
# st.table(df_unique_m)

# st.title('merged')
# st.write(show_dataframe(x = 'm'))
# st.table(df_merged)

# sns.distplot(df_train['critical_temp'])
# st.pyplot()


df_merged_head_30 = df_merged.head(30)
df_merged_head_30

st.title(f'Number of rows: {len(df_merged)}')

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

st.plotly_chart(fig)

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