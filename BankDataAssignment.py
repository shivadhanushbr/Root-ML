#!/usr/bin/env python
# coding: utf-8

# # Bank Data

# ## All Imports

# In[1]:


import pandas as pd ## Importing the Pandas with aliasing as 'pd'
import matplotlib.pyplot as plt ## Importing the MatPlot library with aliasing as plt to plot graphs on data
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np ## Importing the NumPy library with aliasing as np
from sklearn.preprocessing import LabelEncoder as le 
## Importing Label Encoder for converting Categorical Variables to Numerical
from sklearn.model_selection import train_test_split, cross_val_score
## To split the data into Train and Test data.
from sklearn.tree import DecisionTreeClassifier
## Decision Tree classifier library
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


# ### *Loading Data from Excel to Kernel 

# In[2]:


bankData_base = pd.read_csv("bank_data.csv") 
## Loading the 'csv' file to data frame. (Using Pandas 'pd.read_csv' we loaded the data into pandas data frame)


# ## *Basic Data exploration and insights

# In[3]:


bankData_base.shape ## 'DataFrame.shape' gives you the matix mXn number.


# In[4]:


bankData_base.describe ## 'DataFrame.Describe' gives you the complete view of the data frame


# In[5]:


bankData_base.head()
## 'DataFrame.head' gives you the first 5 records from the table. you can point out the number of records to display by 'DataFrame.head(n).'


# In[6]:


bankData_base.tail()
## 'DataFrame.tail' gives you the last 5 records from the table. you can point out the number of records to display by 'DataFrame.tail(n).'


# ### *Complete Data Visualization (Which is not actually 

# In[7]:


bankData_base.plot(kind="bar")  
## Visualize the compelete data in a bar garph. It will be messy if you are not grouping or soring the data before ploting
plt.show  ## To show the plotted graph in console.


# ### *A Sample variable Visualization

# In[8]:


bankData_base['income'].head(50).plot(kind="bar")
## Visualize the data by sorting only the 'INCOME' dimension and filtering only top 50 records to plot. Now the plot looks bit analysable
plt.show ## To show the plotted graph in console.


# ### *Statistical Inference of Diemnsions of Data

# In[9]:


bankData_base.describe() 
## 'DataFrame.describe()' gives you the statistical analysis of data frame. But this working only on Numerical/Continious Data.
## Gives 'Count,Mean,StdDeviation,Min,Max,1st Quartile, 2nd Quartile, 3rd Quartile' of Numerical Data


# ### *Finding the Null Values in each Variable

# In[10]:


bankData_base.isnull().sum()
## Gives you the count of NULL Values for each column.
## As there are no missing values we will not use any techniques to impute missing values


# ##### From Data we can infer that their are 7 Categorical Variables in the dataset
# ##### So to tackle this we will try to bring them down to Numerical as most of the alogorithms and models prifer to use the numerical values over categorical.
# ##### Here the categories are only boolean except 'Region' variable which also has less categories.

# In[11]:


bankData_continious = bankData_base.copy() 
## Making the copy of the data so original is undisturbed for future reference.
## x = y without y.copy() will make a reference of y to x, so all the updates at x will be reflected at y(Original) also


# ### *Fetching All the Variables list and Numeric Variables list

# In[12]:


num_Features = bankData_continious._get_numeric_data().columns
## Fetching only the numeric data variables from the DataFrame
all_Features = bankData_continious.columns.values.tolist()
## Fetching all the Features/Variables from DataFrame
print(all_Features[1:])
print('\n',num_Features)


# ### *Converting all the Categorical Variables into Numeric or Continious Variables

# In[13]:


for allF in all_Features[1:]: #Array starting from index 1 so to remove ID which can be dropped as all teh values are Unnique
    if allF not in num_Features:
        bankData_continious[allF] = le.fit_transform(bankData_continious[allF],bankData_continious[allF])
        ## 'LabelEncoder.fit_transform' fit the column and transforms the variable to numerical variable. It receives 2 parameter X & Y
        ## Converted array of values will be stored in the same variable
        print(bankData_continious[allF].head(4))


# ## *Dimensionality Reduction

# In[14]:


for allF in all_Features[1:]:
    #if (np.var(bankData_continious[allF])) > 0.5:
    print("Column Name: ", allF.upper(), " has variation of ", np.var(bankData_continious[allF])) 
# we can find the variation of each Variables listed. Varaition of a variable should be high to be able to effect results.
# Its obvious that Boolean Variable will have low variance as it has only two values available.


# In[15]:


bankData_target = bankData_continious['loan'].copy()
# Isolating Target Variable


# In[16]:


bankData_variable = bankData_continious.loc[:,bankData_continious.columns != 'loan'].copy()
bankData_indVars = bankData_variable.drop(['id'], axis = 1).copy()
# Isolating All independent variables and dropping of the ID.
# As ID is a variable which has all unique values and will have bad variance.


# In[17]:


bankData_indVars.corr()
# Finding the Correlation among the Independent Variables. 
# Variables with High Correlation among themself shall be given a high look for elimination


# ##### From the results we can identify that 'age' vs 'income' has a correlation of 0.752726 which is a good number to be considered to drop down.
# ##### But by Domain knowledge we know that both 'INCOME' and 'AGE' are important variables for loan prediction.

# In[18]:


for allF in bankData_indVars:
    #print(bankData_indVars[allF].head(2), "\n")
    print("\n Correlation with ", allF, " and Target variable is: ", np.corrcoef(bankData_indVars[allF], bankData_target))


# In[19]:


bankData_indVars.head()


# In[20]:


X_train,X_test,Y_train,Y_test= train_test_split(bankData_indVars, bankData_target, test_size = 0.25, random_state = 20)


# In[21]:


decision_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 93, min_samples_leaf=4, min_samples_split=5, max_features=6)
decision_tree.fit(X_train, Y_train)
Predicted_Results = decision_tree.predict(X_test)


# In[22]:


print(metrics.confusion_matrix(Y_test, Predicted_Results))
print(metrics.accuracy_score(Y_test, Predicted_Results))


# In[23]:


decision_tree = DecisionTreeClassifier(random_state = 0, min_samples_leaf=6, max_features=9, max_leaf_nodes=18)
decision_tree.fit(X_train, Y_train)
Predicted_Results = decision_tree.predict(X_test)
decision_tree.feature_importances_
print(dict(zip(bankData_indVars.columns, decision_tree.feature_importances_)))


# In[ ]:


print(metrics.confusion_matrix(Y_test, Predicted_Results))
print(metrics.accuracy_score(Y_test, Predicted_Results))


# In[ ]:


bankData_indV = bankData_indVars.drop(['current_acc', 'age','sex', 'region', 'car'], axis = 1)


# In[ ]:


X_train1,X_test1,Y_train1,Y_test1= train_test_split(bankData_indV, bankData_target, test_size = 0.25, random_state = 20)


# In[ ]:


decision_tree = DecisionTreeClassifier(random_state = 0, min_samples_leaf=6, max_leaf_nodes=18)
decision_tree.fit(X_train1, Y_train1)
Predicted_Results = decision_tree.predict(X_test1)


# In[ ]:


print(metrics.confusion_matrix(Y_test1, Predicted_Results))
print(metrics.accuracy_score(Y_test1, Predicted_Results)*100)


# In[ ]:


TestRes = decision_tree.predict(X_train1)
print(metrics.accuracy_score(Y_train1, TestRes)*100)


# In[ ]:


randomForest = RandomForestClassifier(n_estimators=20,random_state=20, min_samples_leaf=4, max_leaf_nodes=20, min_samples_split=5)
randomForest.fit(X_train1, Y_train1)
Predicted_RandomResults = randomForest.predict(X_test1)


# In[ ]:


print(metrics.confusion_matrix(Y_test1, Predicted_RandomResults))
print(metrics.accuracy_score(Y_test1, Predicted_RandomResults)*100)


# In[ ]:


TestResRand = randomForest.predict(X_train1)
print(metrics.confusion_matrix(Y_train1, TestResRand))
print(metrics.accuracy_score(Y_train1, TestResRand)*100)


# In[ ]:


metrics.roc_auc_score(Y_test1,Predicted_RandomResults)


# In[ ]:


print(metrics.classification_report(Y_test1,Predicted_RandomResults))


# In[ ]:


metrics.accuracy_score(Y_test1,Predicted_RandomResults)


# In[ ]:


metrics.zero_one_loss(Y_test1,Predicted_RandomResults)


# In[ ]:


metrics.precision_score(Y_test1,Predicted_RandomResults)


# In[ ]:


print((metrics.confusion_matrix(Y_train1, TestResRand))[0][0])


# In[ ]:


print('Sensitity:',(((metrics.confusion_matrix(Y_train1, TestResRand))[1][1])/((metrics.confusion_matrix(Y_train1, TestResRand))[1][1] + (metrics.confusion_matrix(Y_train1, TestResRand))[0][1]))*100)


# In[ ]:


print('Specificity:',(((metrics.confusion_matrix(Y_train1, TestResRand))[0][0])/((metrics.confusion_matrix(Y_train1, TestResRand))[0][0] + (metrics.confusion_matrix(Y_train1, TestResRand))[1][0]))*100)


# In[ ]:




