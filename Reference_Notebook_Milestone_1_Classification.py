#!/usr/bin/env python
# coding: utf-8

# # **Milestone 1**

# ##<b>Problem Definition</b>
# **The context:** Why is this problem important to solve?<br>
# **The objectives:** What is the intended goal?<br>
# **The key questions:** What are the key questions that need to be answered?<br>
# **The problem formulation:** What is it that we are trying to solve using data science?
# 
# ## **Data Description:**
# The Home Equity dataset (HMEQ) contains baseline and loan performance information for 5,960 recent home equity loans. The target (BAD) is a binary variable that indicates whether an applicant has ultimately defaulted or has been severely delinquent. This adverse outcome occurred in 1,189 cases (20 percent). 12 input variables were registered for each applicant.
# 
# 
# * **BAD:** 1 = Client defaulted on loan, 0 = loan repaid
# 
# * **LOAN:** Amount of loan approved.
# 
# * **MORTDUE:** Amount due on the existing mortgage.
# 
# * **VALUE:** Current value of the property. 
# 
# * **REASON:** Reason for the loan request. (HomeImp = home improvement, DebtCon= debt consolidation which means taking out a new loan to pay off other liabilities and consumer debts) 
# 
# * **JOB:** The type of job that loan applicant has such as manager, self, etc.
# 
# * **YOJ:** Years at present job.
# 
# * **DEROG:** Number of major derogatory reports (which indicates a serious delinquency or late payments). 
# 
# * **DELINQ:** Number of delinquent credit lines (a line of credit becomes delinquent when a borrower does not make the minimum required payments 30 to 60 days past the day on which the payments were due). 
# 
# * **CLAGE:** Age of the oldest credit line in months. 
# 
# * **NINQ:** Number of recent credit inquiries. 
# 
# * **CLNO:** Number of existing credit lines.
# 
# * **DEBTINC:** Debt-to-income ratio (all your monthly debt payments divided by your gross monthly income. This number is one way lenders measure your ability to manage the monthly payments to repay the money you plan to borrow.

# ## <b>Important Notes</b>
# 
# - This notebook can be considered a guide to refer to while solving the problem. The evaluation will be as per the Rubric shared for each Milestone. Unlike previous courses, it does not follow the pattern of the graded questions in different sections. This notebook would give you a direction on what steps need to be taken in order to get a viable solution to the problem. Please note that this is just one way of doing this. There can be other 'creative' ways to solve the problem and we urge you to feel free and explore them as an 'optional' exercise. 
# 
# - In the notebook, there are markdowns cells called - Observations and Insights. It is a good practice to provide observations and extract insights from the outputs.
# 
# - The naming convention for different variables can vary. Please consider the code provided in this notebook as a sample code.
# 
# - All the outputs in the notebook are just for reference and can be different if you follow a different approach.
# 
# - There are sections called **Think About It** in the notebook that will help you get a better understanding of the reasoning behind a particular technique/step. Interested learners can take alternative approaches if they want to explore different techniques. 

# ### **Import the necessary libraries**

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
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


# ### **Read the dataset**

# In[50]:


hm=pd.read_csv("hmeq.csv")


# In[51]:


# Copying data to another variable to avoid any changes to original data
data=hm.copy()


# ### **Print the first and last 5 rows of the dataset**

# In[52]:


# Display first five rows
# Remove ___________ and complete the code
data.head()


# In[53]:


# Display last 5 rows
# Remove ___________ and complete the code
data.tail()


# ### **Understand the shape of the dataset**

# In[54]:


# Check the shape of the data
# Remove ___________ and complete the code

data.shape


# **Insights ________**

# ### **Check the data types of the columns**

# In[55]:


# Check info of the data
# Remove ___________ and complete the code
data.info()


# **Insights ______________**

# ### **Check for missing values**

# In[56]:


# Analyse missing values - Hint: use isnull() function
# Remove ___________ and complete the code
data.isna().any()
data.isna().sum()


# In[57]:


data.isnull().sum()
# number of null values in the dataframe


# In[58]:


data.duplicated().sum()
# no duplicates


# In[59]:


# Check the percentage of missing values in the each column.
# Hint: divide the result from the previous code by the number of rows in the dataset
# Remove ___________ and complete the code

round((data.isnull().sum()/len(data))*100,2)


# **Insights ________**

# ### **Think about it:**
# - We found the total number of missing values and the percentage of missing values, which is better to consider?
# - What can be the limit for % missing values in a column in order to avoid it and what are the challenges associated with filling them and avoiding them? 

# **We can convert the object type columns to categories**
# 
# `converting "objects" to "category" reduces the data space required to store the dataframe`

# ### **Convert the data types**

# In[60]:


cols = data.select_dtypes(['object']).columns.tolist()

#adding target variable to this list as this is an classification problem and the target variable is categorical

cols.append('BAD')


# In[61]:


cols


# In[62]:


# Changing the data type of object type column to category. hint use astype() function
# remove ___________ and complete the code
for col in ['REASON', 'JOB', 'BAD']:
    data[col] = data[col].astype('category')    


# In[63]:


# Checking the info again and the datatype of different variable
# remove ___________ and complete the code

data.info()


# In[64]:


data['JOB'].unique()
data['JOB'] = data['JOB'].replace('nan','Other')


# In[65]:


data['JOB'].unique()


# ### **Analyze Summary Statistics of the dataset**

# In[66]:


# Analyze the summary statistics for numerical variables
# Remove ___________ and complete the code

data.describe()


# **Insights ______________**

# In[67]:


# Check summary for categorical data - Hint: inside describe function you can use the argument include=['category']
# Remove ___________ and complete the code

data.describe(include=['category']).T


# **Insights _____________**

# **Let's look at the unique values in all the categorical variables**

# In[68]:


# Checking the count of unique values in each categorical column 
# Remove ___________ and complete the code

cols_cat= data.select_dtypes(['category'])

for i in cols_cat.columns:
    print('Unique values in',i, 'are :')
    print(data[i].nunique())
    print('*'*40)


# **Insights _____________**

# ### **Think about it**
# - The results above gave the absolute count of unique values in each categorical column. Are absolute values a good measure? 
# - If not, what else can be used? Try implementing that. 

# ## **Exploratory Data Analysis (EDA) and Visualization**

# ## **Univariate Analysis**
# 
# Univariate analysis is used to explore each variable in a data set, separately. It looks at the range of values, as well as the central tendency of the values. It can be done for both numerical and categorical variables

# ### **1. Univariate Analysis - Numerical Data**
# Histograms and box plots help to visualize and describe numerical data. We use box plot and histogram to analyze the numerical columns.

# In[69]:


# While doing uni-variate analysis of numerical variables we want to study their central tendency and dispersion.
# Let us write a function that will help us create boxplot and histogram for any input numerical variable.
# This function takes the numerical column as the input and return the boxplots and histograms for the variable.
# Let us see if this help us write faster and cleaner code.
def histogram_boxplot(feature, figsize=(15,10), bins = None):
    """ Boxplot and histogram combined
    feature: 1-d feature array
    figsize: size of fig (default (9,8))
    bins: number of bins (default None / auto)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(nrows = 2, # Number of rows of the subplot grid= 2
                                           sharex = True, # x-axis will be shared among all subplots
                                           gridspec_kw = {"height_ratios": (.25, .75)}, 
                                           figsize = figsize 
                                           ) # creating the 2 subplots
    sns.boxplot(feature, ax=ax_box2, showmeans=True, color='violet') # boxplot will be created and a star will indicate the mean value of the column
    sns.distplot(feature, kde=F, ax=ax_hist2, bins=bins,palette="winter") if bins else sns.distplot(feature, kde=False, ax=ax_hist2) # For histogram
    ax_hist2.axvline(np.mean(feature), color='green', linestyle='--') # Add mean to the histogram
    ax_hist2.axvline(np.median(feature), color='black', linestyle='-') # Add median to the histogram


# #### Using the above function, let's first analyze the Histogram and Boxplot for LOAN

# In[70]:


# Build the histogram boxplot for Loan
histogram_boxplot(data['LOAN'])


# **Insights __________**

# #### **Note:** As done above, analyze Histogram and Boxplot for other variables

# **Insights ____________**

# ### **2. Univariate Analysis - Categorical Data**

# In[71]:


# Function to create barplots that indicate percentage for each category.

def perc_on_bar(plot, feature):
    '''
    plot
    feature: categorical feature
    the function won't work if a column is passed in hue parameter
    '''

    total = len(feature) # length of the column
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total) # percentage of each class of the category
        x = p.get_x() + p.get_width() / 2 - 0.05 # width of the plot
        y = p.get_y() + p.get_height()           # height of the plot
        ax.annotate(percentage, (x, y), size = 12) # annotate the percentage 
        
    plt.show() # show the plot


# #### Analyze Barplot for DELINQ

# In[72]:


#Build barplot for DELINQ

plt.figure(figsize=(15,5))
ax = sns.countplot(data["DELINQ"],palette='winter')
perc_on_bar(ax,data["DELINQ"])


# **Insights ________**

# #### **Note:** As done above, analyze Histogram and Boxplot for other variables.

# **Insights _____________**

# ## **Bivariate Analysis**

# ###**Bivariate Analysis: Continuous and Categorical Variables**

# #### Analyze BAD vs Loan

# In[73]:


sns.boxplot(data["BAD"],data['LOAN'],palette="PuBu")


# **Insights ______**

# ####**Note:** As shown above, perform Bi-Variate Analysis on different pair of Categorical and continuous variables

# ### **Bivariate Analysis: Two Continuous Variables**

# In[74]:


sns.scatterplot(data["VALUE"],data['MORTDUE'],palette="PuBu")


# **Insights: _____**

# #### **Note:** As shown above, perform Bivariate Analysis on different pairs of continuous variables

# **Insights ____________**

# ### **Bivariate Analysis:  BAD vs Categorical Variables**

# **The stacked bar chart (aka stacked bar graph)** extends the standard bar chart from looking at numeric values across one categorical variable to two.

# In[75]:


### Function to plot stacked bar charts for categorical columns

def stacked_plot(x):
    sns.set(palette='nipy_spectral')
    tab1 = pd.crosstab(x,data['BAD'],margins=True)
    print(tab1)
    print('-'*120)
    tab = pd.crosstab(x,data['BAD'],normalize='index')
    tab.plot(kind='bar',stacked=True,figsize=(10,5))
    plt.legend(loc='lower left', frameon=False)
    plt.legend(loc="upper left", bbox_to_anchor=(1,1))
    plt.show()


# #### Plot stacked bar plot for for LOAN and REASON

# In[76]:


# Plot stacked bar plot for BAD and REASON
stacked_plot(data['REASON'])


# **Insights ____________**

# #### **Note:** As shown above, perform Bivariate Analysis on different pairs of Categorical vs BAD

# **Insights ___________________**

# ### **Multivariate Analysis**

# #### Analyze Correlation Heatmap for Numerical Variables

# In[77]:


# Separating numerical variables
numerical_col = data.select_dtypes(include=np.number).columns.tolist()

# Build correlation matrix for numerical columns
# Remove ___________ and complete the code

corr = data[numerical_col].corr()

# plot the heatmap
# Remove ___________ and complete the code

plt.figure(figsize=(16,12))
sns.heatmap(corr,cmap='coolwarm',vmax=1,vmin=-1,
        fmt=".2f",
        xticklabels=corr.columns,
        yticklabels=corr.columns);


# In[78]:


# Build pairplot for the data with hue = 'BAD'
# Remove ___________ and complete the code

sns.pairplot(data,hue='BAD')


# ### **Think about it**
# - Are there missing values and outliers in the dataset? If yes, how can you treat them? 
# - Can you think of different ways in which this can be done and when to treat these outliers or not?
# - Can we create new features based on Missing values?

# #### Treating Outliers

# In[79]:


def treat_outliers(df,col):
    '''
    treats outliers in a varaible
    col: str, name of the numerical varaible
    df: data frame
    col: name of the column
    '''   
    Q1=df[col].quantile(0.25) # 25th quantile
    Q3=df[col].quantile(0.75)  # 75th quantile
    IQR= Q3-Q1   # IQR Range
    Lower_Whisker =  Q1 - 1.5*IQR  #define lower whisker
    Upper_Whisker = Q3 + 1.5*IQR  # define upper Whisker
    df[col] = np.clip(df[col], Lower_Whisker, Upper_Whisker) # all the values samller than Lower_Whisker will be assigned value of Lower_whisker 
                                                            # and all the values above upper_whishker will be assigned value of upper_Whisker 
    return df

def treat_outliers_all(df, col_list):
    '''
    treat outlier in all numerical varaibles
    col_list: list of numerical varaibles
    df: data frame
    '''
    for c in col_list:
        df = treat_outliers(df,c)
        
    return df
    


# In[80]:


df_raw = data.copy()

numerical_col = df_raw.select_dtypes(include=np.number).columns.tolist()# getting list of numerical columns

df = treat_outliers_all(df_raw,numerical_col)


# #### Adding new columns in the dataset for each column which has missing values 

# In[81]:


#For each column we create a binary flag for the row, if there is missing value in the row, then 1 else 0. 
def add_binary_flag(df,col):
    '''
    df: It is the dataframe
    col: it is column which has missing values
    It returns a dataframe which has binary falg for missing values in column col
    '''
    new_col = str(col)
    new_col += '_missing_values_flag'
    df[new_col] = df[col].isna()
    return df


# In[82]:


# list of columns that has missing values in it
missing_col = [col for col in df.columns if df[col].isnull().any()]

for colmn in missing_col:
    add_binary_flag(df,colmn)
    


# In[83]:


df


# #### Filling missing values in numerical columns with median and mode in categorical variables

# In[99]:


#  Treat Missing values in numerical columns with median and mode in categorical variables
# Select numeric columns.
num_data = data.select_dtypes('number').columns.tolist()

# Select string and object columns.
cat_data = data.select_dtypes('category').columns.tolist()#df.select_dtypes('object')

# Fill numeric columns with median.
for col in num_data:
    data[col] = data[col].fillna(data[col].median())

# Fill object columns with model.
# Remove _________ and complete the code
for column in cat_data:
    mode = data[column].mode()[0]
    data[column] = data[column].fillna(mode)


# In[98]:


data


# In[103]:


data


# ## **Proposed approach**
# **1. Potential techniques** - What different techniques should be explored?
# 
# **2. Overall solution design** - What is the potential solution design?
# 
# **3. Measures of success** - What are the key measures of success?
