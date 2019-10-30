#!/usr/bin/env python
# coding: utf-8

# Imports all the libraries at one place to have better look at the list of the Libraries

# In[7]:


#EDA Libraries
import pandas as pd
import numpy as np
import pandas_profiling as pp
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer,quantile_transform
#Models
from sklearn.ensemble import RandomForestClassifier,GradientBoostingRegressor,RandomForestRegressor
from sklearn import linear_model as lm
import xgboost as xgb
from sklearn import metrics


# Read the data

# In[8]:


dFrame = pd.read_csv("C:/Users/shivu/Desktop/DigiSpice/Assignment_Data_Science/assignment_train.csv")


# #### Lets Start with EDA

# Understanding the variables with oe magical library 'Pandas Profile'

# In[9]:


profile = pp.ProfileReport(dFrame)
#profile.to_file(output_file="C:/Users/shivu/Desktop/DigiSpice/Assignment_Data_Science/output.html")
profile


# In[10]:


#profile.get_rejected_variables() gives you the list of variables identified as reject by 'Pandas Profile library' by 
#correlation factor

#I have listed them out externally from previious Pandas Profile run as running pandas profile is computationally expensive.
removableVariables = ['3M_all_max', '3M_weekly_all_avg', 'all_gtv_last10weeks_w1', 'all_gtv_last10weeks_w10', 'all_gtv_last10weeks_w2', 'all_gtv_last10weeks_w3', 'all_gtv_last10weeks_w4', 'all_gtv_last10weeks_w5', 'all_gtv_last10weeks_w6', 'all_gtv_last10weeks_w7', 'all_gtv_last10weeks_w8', 'all_gtv_last10weeks_w9', 'all_gtv_last12Months_m1', 'all_gtv_last12Months_m10', 'all_gtv_last12Months_m11', 'all_gtv_last12Months_m2', 'all_gtv_last12Months_m3', 'all_gtv_last12Months_m4', 'all_gtv_last12Months_m5', 'all_gtv_last12Months_m6', 'all_gtv_last12Months_m7', 'all_gtv_last12Months_m8', 'all_gtv_last12Months_m9', 'all_highest_seg_last6M', 'all_lst30days_vsmean_lst3m', 'all_mrr_vsmax_lst3mnth', 'all_mrr_vsmean_lst3mnth', 'all_mtd_vs_lstmtd', 'all_mtd_vs_max_lst3M', 'all_mtd_vs_mean_lst3M', 'all_mtd_vs_min_lst3M', 'all_seg']
dFrame1 = dFrame[dFrame.columns.difference(removableVariables)]


# Let us understand the Target Variable

# In[11]:


#We already have the necessary stats of the target variable but again trying to list out for more understanding
get_ipython().run_line_magic('matplotlib', 'inline')
print("Skewness: ",dFrame1["business_risk"].skew())
print("Kurtosis value(understanding tails) :",dFrame1["business_risk"].kurtosis())
print("Missing Values: ", dFrame1["business_risk"].isna().sum())

sb.set(color_codes=True)
sb.distplot(dFrame1["business_risk"])
plt.show()


# Above Data shows Target has much more tail length. Graph and Kurtosis values shows the same.

# In[12]:


# looking for outliers
q75, q25 = np.percentile(dFrame1["business_risk"], [75 ,25])
iqr = q75 - q25
print("Outliers count with IQR*1.2 =", dFrame1[dFrame1["business_risk"] > (iqr*1.2)].shape[0])


# In[13]:


#Let explore outlier data if its good to be removed or we have keep the data as we are loosing info.
pp.ProfileReport(dFrame1[dFrame1["business_risk"] > (iqr*1.0)])


# Outlier data is also well diverse/has good variances for features. So we can't remove outliers as we may loose info hence we do capping.
# 

# In[14]:


# Trying to reduce Kurtosis values and skewness by 'CAPPING' the outliers
nonCapDF=dFrame1[dFrame1["business_risk"] < (iqr*1.2)].copy()
toCapDF=dFrame1[dFrame1["business_risk"] > (iqr*1.2)].copy()
print("Max value of Target Variable to be used to Cap all the values above iqr*1 is: ",max(nonCapDF["business_risk"]))
toCapDF["business_risk"] = max(nonCapDF["business_risk"])


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
dFrame2 = pd.concat([nonCapDF,toCapDF])
print("Skewness after Capping: ",dFrame2["business_risk"].skew())
print("Kurtosis value(understanding tails) after Capping :",dFrame2["business_risk"].kurtosis())

sb.set(color_codes=True)
sb.distplot(dFrame2["business_risk"])
plt.show()


# In[16]:


# We will try to Normalize the Target Variables so it satifies the Linear regression assumptions


# uniqueValues, occurCount = np.unique(dFrame1["business_risk"], return_counts=True)
# dic = {}
# for i, con in enumerate(uniqueValues):
#     dic[con] = occurCount[i]
# dic

# In[17]:


Target = dFrame2["business_risk"]
Tmm = [] #Target Min Max Normalization
Tm = [] # Target Mean Normalization
Tf = [] # Target F-Score Normalization
m = Target.mean()
mx = max(Target)
mn = min(Target)
s = Target.std()
for val in Target:
    a=[(val - mn)/(mx-mn)]
    Tmm.append(a)
    b=[(val - m)/(mx-mn)]
    Tm.append(b)
    c=[(val - m)/s]
    Tf.append(c)
#sb.distplot(Target)


# In[18]:


sb.distplot(Tmm)
sb.distplot(Tm)
sb.distplot(Tf)


# Form Above graphs: Each way of normalization didnt yield any satisfactory results

# In[19]:


# we will try to apply quantile transform to bring down the target variable to normal distribution
y_transMM = quantile_transform(Tmm,
                             n_quantiles=20000,
                             output_distribution='normal',
                             copy=True)
y_transM = quantile_transform(Tm,
                             n_quantiles=20000,
                             output_distribution='normal',
                             copy=True)
y_transF = quantile_transform(Tf,
                             n_quantiles=20000,
                             output_distribution='normal',
                             copy=True)
sb.distplot(y_transMM)
sb.distplot(y_transM)
sb.distplot(y_transF)


# With all the possible try we are not able to Normalize the Target Variable.

# #### We will choose non linear models to fit the training data

# Before modeling we will try to do Dimensionality reduction. We already have done the feature reduction after the results of the 
# pandas profiling. But still we are left with 62 columns which is still a big number of features. 
# RandomForestClassifier,GradientBoostingRegressor,RandomForestRegressor

# In[20]:


inputVariables = dFrame2.drop(["business_risk", "agent_id"], axis =1)
TargetVariable = dFrame2["business_risk"]


# In[21]:


RFreg = RandomForestRegressor(max_depth=3, random_state=0,n_estimators=100)
RFreg.fit(inputVariables, TargetVariable)
importances = RFreg.feature_importances_


# In[22]:


impVariables = []
for i, var in enumerate(inputVariables):
    if importances[i]>0:
        #print(var," importance: ", importances[i])
        impVariables.append(var)


# In[23]:


#toRemove = inputVariables.columns-impVariables
toRemove = []
for ival in inputVariables:
    if ival not in impVariables:
        toRemove.append(ival)
toRemove
impInputVars = inputVariables.drop(toRemove, axis=1)


# In[24]:


ixTrain, ixTest, iyTrain, iyTest = train_test_split(impInputVars,TargetVariable, test_size=0.25, random_state=20)
xTrain, xTest, yTrain, yTest = train_test_split(inputVariables,TargetVariable, test_size=0.25, random_state=20)


# #### Random Forest Regressor Model

# In[25]:


RFregrs = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=8,
           max_features=15, max_leaf_nodes=10,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=3, min_samples_split=5,
           min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=None,
           oob_score=False, random_state=40, verbose=0, warm_start=False)


# In[26]:


RFregrs.fit(ixTrain,iyTrain)
RFregrs.predict(ixTrain)
RFregrs.score(ixTrain,iyTrain)


# #### Generalized Linear Model-Lasso

# In[27]:


LMlasso = lm.Lasso(alpha=0.2, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=40,
   selection='cyclic', tol=0.0001, warm_start=False)


# In[28]:


LMlasso.fit(ixTrain,iyTrain)
LMlasso.predict(ixTrain)
LMlasso.score(ixTrain,iyTrain)


# #### Gradient Boosting Regression

# In[29]:


GBM = GradientBoostingRegressor(loss='huber', alpha=0.45,subsample=0.8,
                                n_estimators=800, max_depth=6,
                                learning_rate=.5, min_samples_leaf=4,
                                min_samples_split=6, validation_fraction=0.1, n_iter_no_change =500)


# In[30]:


GBM.fit(ixTrain,iyTrain)
GBM.predict(ixTrain)
GBM.score(ixTrain,iyTrain)


# #### Extreme Gradient Boosting Regression

# In[31]:


XGB = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.5, max_delta_step=2,
       max_depth=6, min_child_weight=4, missing=None, n_estimators=350,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=25,alpha = 2,
       tree_method= 'approx',reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
       silent=True, subsample=0.8)


# In[32]:


#Training fit and accuracy
XGB.fit(ixTrain,iyTrain)
XGB.score(ixTrain,iyTrain)


# In[33]:


# Testing accuracy
XGB.score(ixTest,iyTest)


# In[34]:



print("Explained Variance: ",metrics.explained_variance_score(iyTrain,XGB.predict(ixTrain)))
print("Max error: ",metrics.max_error(iyTrain,XGB.predict(ixTrain)))
print("MAE: ", metrics.mean_absolute_error(iyTrain,XGB.predict(ixTrain)))
print("MSE: ", metrics.mean_squared_error(iyTrain,XGB.predict(ixTrain)))
print("R2: ", metrics.r2_score(iyTrain,XGB.predict(ixTrain)))


# In[35]:


print("Explained Variance: ",metrics.explained_variance_score(iyTest,XGB.predict(ixTest)))
print("Max error: ",metrics.max_error(iyTest,XGB.predict(ixTest)))
print("MAE: ", metrics.mean_absolute_error(iyTest,XGB.predict(ixTest)))
print("MSE: ", metrics.mean_squared_error(iyTest,XGB.predict(ixTest)))
print("R2: ", metrics.r2_score(iyTest,XGB.predict(ixTest)))


# In[39]:


testDF = pd.read_csv("C:/Users/shivu/Desktop/DigiSpice/Assignment_Data_Science/assignment_test.csv")
testDF.shape


# In[42]:


removableVariables = ['3M_all_max', '3M_weekly_all_avg', 'all_gtv_last10weeks_w1', 'all_gtv_last10weeks_w10', 'all_gtv_last10weeks_w2', 'all_gtv_last10weeks_w3', 'all_gtv_last10weeks_w4', 'all_gtv_last10weeks_w5', 'all_gtv_last10weeks_w6', 'all_gtv_last10weeks_w7', 'all_gtv_last10weeks_w8', 'all_gtv_last10weeks_w9', 'all_gtv_last12Months_m1', 'all_gtv_last12Months_m10', 'all_gtv_last12Months_m11', 'all_gtv_last12Months_m2', 'all_gtv_last12Months_m3', 'all_gtv_last12Months_m4', 'all_gtv_last12Months_m5', 'all_gtv_last12Months_m6', 'all_gtv_last12Months_m7', 'all_gtv_last12Months_m8', 'all_gtv_last12Months_m9', 'all_highest_seg_last6M', 'all_lst30days_vsmean_lst3m', 'all_mrr_vsmax_lst3mnth', 'all_mrr_vsmean_lst3mnth', 'all_mtd_vs_lstmtd', 'all_mtd_vs_max_lst3M', 'all_mtd_vs_mean_lst3M', 'all_mtd_vs_min_lst3M', 'all_seg']
testdFrame1 = testDF[testDF.columns.difference(removableVariables)]
ttoRemove = []
for ival in inputVariables:
    if ival not in impVariables:
        ttoRemove.append(ival)
timpInputVars = inputVariables.drop(ttoRemove, axis=1)


# In[43]:


XGB.predict(timpInputVars)

