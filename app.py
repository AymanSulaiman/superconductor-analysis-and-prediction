import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import os
import seaborn as sns

import time

st.title('Superconductor Analysis and Machine Learning')


path_merged = os.path.join('merged.csv')
path_train = os.path.join('data','train.csv')
path_unique_m = os.path.join('data','unique_m.csv')

df_merged = pd.read_csv(path_merged)
df_train = pd.read_csv(path_train)
df_unique_m = pd.read_csv(path_unique_m)


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


x = df_merged.head(30)
x

st.write('''
# hello world

This is a blah blah blah.
''')