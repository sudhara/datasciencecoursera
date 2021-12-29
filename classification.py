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
import seaborn as sns

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

data=pd.read_csv("hmeq.csv")
# Create an instance of the app

st.title("Classification Problem")
st.header('Data Exploration')

st.write(data)

col1 = st.columns(1)
col3, col4 = st.columns(2)
col5 = st.columns(1)

data['JOB'] = data['JOB'].fillna('Other')
cols = data.select_dtypes(['object']).columns.tolist()

for i in cols:
    data[i] = data[i].astype('category')

st.sidebar.title("Select Visual Charts")
st.sidebar.markdown("Select the Charts/Plots accordingly:")
chart_visual = st.sidebar.selectbox('Select Charts/Plot type', ('Line Chart', 'Bar Chart', 'Box Chart', 'Histogram'))

st.sidebar.checkbox("Show Analysis by Occupation", True, key=1)
selected_job = st.sidebar.selectbox('Select Occupation', options=data['JOB'].unique())

if st.sidebar.button("Generate"):
    if chart_visual == 'Box Chart':
        st.write(sns.boxplot(x="JOB", y="LOAN", palette="husl", data=data))
        st.pyplot()

    if chart_visual == 'Bar Chart':
        data_bar = pd.DataFrame(data['JOB'] == selected_job)
        st.write(sns.barplot(data_bar, y = data_bar.groupby('YOJ').count(), x=data_bar['YOJ'].unique()))
        st.pyplot()

    if chart_visual == 'Line Chart':
        data_line = pd.DataFrame(data['JOB'] == selected_job)
        st.write(sns.lineplot(data_line,y='MORTDUE'))
        st.pyplot()

    if chart_visual== 'Histogram':
        data_hist = pd.DataFrame(data[:],columns=['BAD','LOAN','MORTDUE'])
        st.write(sns.histplot(data=data_hist, x='LOAN', kde=True))
        st.pyplot()
        st.write(sns.histplot(data=data_hist, x='MORTDUE', kde=True))
        st.pyplot()
