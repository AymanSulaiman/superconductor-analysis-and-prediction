import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import os

import time

st.title('Superconductor Analysis and Machine Learning')


path = os.path.join('merged.csv')
df = pd.read_csv(path)
df
