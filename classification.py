import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score,precision_score,recall_score,f1_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as stats
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import plotly.express as go

data=pd.read_csv("hmeq.csv")

st.title("Classification Problem")
st.header('Data Exploration')
st.write(data)

data['JOB'] = data['JOB'].fillna('Other')
cols = data.select_dtypes(['object']).columns.tolist()

for i in cols:
    data[i] = data[i].astype('category')

st.sidebar.title("Select Visual Charts")
st.sidebar.markdown("Select the Charts/Plots accordingly:")
chart_visual = st.sidebar.selectbox('Select Charts/Plot type', ('Line Chart', 'Bar Chart', 'Box Chart'))

st.sidebar.checkbox("Show Analysis by Occupation", True, key=1)
selected_status = st.sidebar.selectbox('Select Occupation', options=data['JOB'].nunique())

fig = go.Figure()
if chart_visual == 'Box Chart':
    fig.add_trace(go.box(data, x="JOB", y="LOAN"))
