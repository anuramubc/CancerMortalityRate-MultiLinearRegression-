#!/usr/bin/env python
# coding: utf-8

# Multiple Linear Regression Challenge
# Your Task: Build a multivariate Ordinary Least Squares regression model to predict "TARGET_deathRate"

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Checking the encoding of the dataset
# Use package called "chardet"

# In[ ]:


import chardet
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result


# Importing the dataset

# In[21]:


file = '/Users/anuram/Library/Mobile Documents/com~apple~CloudDocs/Linear Regression /Cancer datset/cancer_reg.csv'
df = pd.read_csv(file, encoding='ISO-8859-1')


# Data summary and description
# df has 3047 rows and 34 columns
# column names of the dataframe is ['avgAnnCount', 'avgDeathsPerYear', 'TARGET_deathRate', 'incidenceRate',
#        'medIncome', 'popEst2015', 'povertyPercent', 'studyPerCap', 'binnedInc',
#        'MedianAge', 'MedianAgeMale', 'MedianAgeFemale', 'Geography',
#        'AvgHouseholdSize', 'PercentMarried', 'PctNoHS18_24', 'PctHS18_24',
#        'PctSomeCol18_24', 'PctBachDeg18_24', 'PctHS25_Over',
#        'PctBachDeg25_Over', 'PctEmployed16_Over', 'PctUnemployed16_Over',
#        'PctPrivateCoverage', 'PctPrivateCoverageAlone', 'PctEmpPrivCoverage',
#        'PctPublicCoverage', 'PctPublicCoverageAlone', 'PctWhite', 'PctBlack',
#        'PctAsian', 'PctOtherRace', 'PctMarriedHouseholds', 'BirthRate']

# In[22]:


df.shape
df.head()
df.columns
df.describe()


# Check for Missing values
# 
# There are missing values in 
# 1. PctSomeCol18_24: Percent of county residents ages 18-24 highest education attained: some college
# 2. PctEmployed16_Over: Percent of county residents ages 16 and over employed
# 3. PctPrivateCoverageAlone: Percent of county residents with private health coverage alone (no public assistance) 

# In[31]:


plt.figure(figsize=(12,4))
sns.heatmap(df.isnull(),cbar=False,cmap='viridis',yticklabels=False)
plt.title('Missing value in the dataset');


# In[30]:


# Use missing no package to visualize NaN (missing) values
import missingno as msno
msno.matrix(df)


# Correlation plot
# There are some correlations between the features and the target
# 
# The top three features that have high correlation with the target_deathrate are:
# incidenceRate              0.449432
# PctPublicCoverageAlone     0.449358
# povertyPercent             0.429389
# 

# In[86]:


# correlation plot
#correlation heat map
corr = df.corr()
sns.heatmap(corr,annot= False);


# In[37]:


#correlation heat map

def correlation_heatmap(df):
    correlations = df.corr()

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=False, cbar_kws={"shrink": .70})
    plt.show();
    
correlation_heatmap(df)


# In[42]:


corr['TARGET_deathRate'].sort_values(ascending = False)


# Deal with Missing data
# 1. DROPPING
# 2. IMPUTATE (Missing values in continuous data can be solved by imputing with mean ,median ,mode or with multiple imputation)
# 
# Using mean values to fill the missing values in PctSomeCol18_24,PctEmployed16_Over,PctPrivateCoverageAlone

# In[29]:


df['PctSomeCol18_24'].fillna(df['PctSomeCol18_24'].mean(),inplace=True)
df['PctEmployed16_Over'].fillna(df['PctEmployed16_Over'].mean(),inplace=True)
df['PctPrivateCoverageAlone'].fillna(df['PctPrivateCoverageAlone'].mean(),inplace=True)


# Distribution of the target death rate 
# - Looks like a right skewed distribution

# In[79]:


ax = sns.histplot(df['TARGET_deathRate'],bins=50,color='r', kde=False)
ax.set_title('Distribution of death rate')


# Splitting data into train and test data

# In[87]:


from sklearn.model_selection import train_test_split
X = df[['incidenceRate','PctPublicCoverageAlone','povertyPercent']] # Independet variables
y = df['TARGET_deathRate'] # dependent variable

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=23)


# In[114]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
fit = lm.fit(X_train,y_train)
fit.summary
R_square_train= fit.score(X_train,y_train)
R_square_train


# The coefficients of the multiple linear model:
# Y = theta_0 + theta_1*incidenceRate + theta_2*PctPublicCoverageAlone+ theta_3*povertyPercent
# 
# where 
# theta_0 = 37.74
# theta_1 = 0.227
# theta_2 = 1.299
# theta_3 = 0.819
# 

# In[103]:


sk_theta = [fit.intercept_]+list(fit.coef_)
sk_theta


# In[104]:


parameter = ['theta_'+str(i) for i in range(1,X_train.shape[1]+1)]
parameter = ['theta_0'] + list(parameter)
columns = ['intersect:x_0=1'] + list(X.columns.values)
parameter_df = pd.DataFrame({'Parameter':parameter,'Columns':columns,'theta':sk_theta})
parameter_df


# Evaluation of the model
# 
# - Huge difference between actual and predicted values. (Maybe these features can't explain the output clearly
# - R^2 of the model on test data is 37%. Maybe fit another model?
# - RMSE is quite high. More

# In[110]:


# predicting output from the test data
y_pred = fit.predict(X_test)


# In[109]:


# data frame comparing the results from predicted y and actual y in test set
predict_df = pd.DataFrame({'Actual' : y_test, 'Predicted': y_pred})
predict_df['Diff'] = predict_df['Actual'] - predict_df['Predicted']
predict_df.reset_index(drop = True, inplace = True)


# In[116]:


#Evaluvation: MSE
from sklearn.metrics import mean_squared_error
J_mse_test = mean_squared_error(y_pred, y_test)
J_rmse_test = np.sqrt(J_mse_test)
# R_square
R_square_test = fit.score(X_test,y_test)
print('The Root Mean Square Error(RMSE) or J(theta) is: ',J_rmse_test)
print('R square obtain for scikit learn library is :',R_square_test)


# The residuals are linear but the homoscedacity assumption fails. Since tthe variances are not randomly distributed.

# In[121]:


# Check for Multivariate Normality
# Quantile-Quantile plot 
f,ax = plt.subplots(1,2,figsize=(14,6))
import scipy as sp
_,(_,_,r)= sp.stats.probplot((y_test - y_pred),fit=True,plot=ax[0])
ax[0].set_title('Check for Multivariate Normality: \nQ-Q Plot')

#Check for Homoscedasticity
sns.scatterplot(y = (y_test - y_pred), x= y_pred, ax = ax[1],color='r') 
ax[1].set_title('Check for Homoscedasticity: \nResidual Vs Predicted');


# The data is not linear, it has a cluster of points and possibly an outlier point in the data around (350).

# In[122]:


# Check for Linearity
f = plt.figure(figsize=(14,5))
ax = f.add_subplot(121)
sns.scatterplot(y_test,y_pred,ax=ax,color='r')
ax.set_title('Check for Linearity:\n Actual Vs Predicted value')

# Check for Residual normality & mean
ax = f.add_subplot(122)
sns.distplot((y_test - y_pred),ax=ax,color='b')
ax.axvline((y_test - y_pred).mean(),color='k',linestyle='--')
ax.set_title('Check for Residual normality & mean: \n Residual eror');


# In[123]:


# Check for Multicollinearity
#Variance Inflation Factor
VIF = 1/(1- R_square_test)
VIF


# In[ ]:




