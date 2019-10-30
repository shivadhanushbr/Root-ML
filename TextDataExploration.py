#!/usr/bin/env python
# coding: utf-8

# We will import some important Libraries that will be used in this assignment.

# In[2]:


import numpy as np  # A number Python library for data manupliation 
import pandas as pd  # A library to manage the Dataframe and its operations
import os  # A library to carry operations on system 
import random
import time  # A library to keep the time stamp
from sklearn.model_selection import train_test_split  # A library from Sklearn under model selection to split the data
from matplotlib import pyplot as plt  # A library from matplotlib to plot the data in a 2D graph 
from sklearn.preprocessing import LabelEncoder, Normalizer # A library from Sklearn under Preprocessingto Normalize the data
import seaborn as sb
import nltk
from sklearn.cluster import KMeans
from array import *


# Reading the Data from the data csv file

# In[3]:


dFrame = pd.read_csv("data.csv")


# Exploring the data

# In[4]:


dFrame.shape


# In[5]:


dFrame.dtypes


# In[6]:


dFrame.head(10)


# Lets Normalize the possible data.
# 1. Salary frequency

# In[7]:


#Salary frequency says about the salary frequency payment to the employee by employeer 
# here it has 3 different modes. We will try to normalize it by reducing the modes to 1.
# By converting the 'Hourly' and 'Daily' to 'Annualy' will solve the issue. But we have to normalize the its related data in 
# 'salary range from' and 'salary range to' to annual salary number as well. 


# In[8]:


dFrame["Salary Frequency"].value_counts()


# In[9]:


#Total Number of hours per day = 8 hours
#Total number of days per week = 5 days
#Total number of Days per year = 5 * 52 (52 weeks per year) = 260
#Total hours = 8 * 260 = 2080

#Total number of working days per year = 5 * 52 = 260


# In[10]:


for i, freq in enumerate(dFrame["Salary Frequency"]):
    if freq == "Hourly":
        dFrame["Salary Range From"].loc[i] = dFrame["Salary Range From"].loc[i] * 2080
        dFrame["Salary Range To"].loc[i] = dFrame["Salary Range To"].loc[i] * 2080
    if freq == "Daily":
        dFrame["Salary Range From"].loc[i] = dFrame["Salary Range From"].loc[i] * 260
        dFrame["Salary Range To"].loc[i]  = dFrame["Salary Range To"].loc[i] * 260


# In[11]:


#Now the Salary Frequency has all the values normalized to 'Annualy' and hence we can remove this column it had no variance.
dFrame = dFrame.drop('Salary Frequency', axis =1)


# 1. Looking for Missing values in each column

# In[12]:


# we are looking for missing data in each column and as per thumbrule we will remove the columns which has more than 30% missing
for col in dFrame.columns:
    if dFrame[col].isnull().sum()*100/dFrame.shape[0] > 0:
        print(col," = ",str(dFrame[col].isnull().sum()*100/dFrame.shape[0]))
    if dFrame[col].isnull().sum()*100/dFrame.shape[0] > 30:
        print("Dropping the columns ",col, " as it has more than 30% of missing values")
        dFrame = dFrame.drop(col , axis=1)
print("-------------------------------------------")
print("Number of columns left after removal = ",str(len(dFrame.columns)))
print("Columns after dropping columns with missing values > 30% : ", dFrame.columns)


# Now we are left with 22 Columns to look for more

# In[13]:


print("Columns left with missing values to be imputed ")
for col in dFrame.columns:
    if dFrame[col].isnull().sum()*100/dFrame.shape[0] > 0:
        print(col," = ",str(dFrame[col].isnull().sum()*100/dFrame.shape[0]), " Count: ", str(dFrame[col].isnull().sum()))


# Imputing for feature "Job Category"

# In[14]:


dFrame["Job Category"].value_counts() 
# Checking each class count to find the class with highest frequency to be imputed over missing values


# In[15]:


dFrame["Job Category"].fillna("Engineering, Architecture, & Planning", inplace=True)
# Filling the null values with the class of higest frequency


# Imputing for "Full-Time/Part-Time indicator"

# In[16]:


dFrame["Full-Time/Part-Time indicator"].value_counts()
# Checking each class count to find the class with highest frequency to be imputed over missing values


# In[17]:


dFrame["Full-Time/Part-Time indicator"].fillna("F", inplace=True)
# Filling the null values with the class of higest frequency


# Imputing for "Preferred Skills"

# I am trying to impute the 'Prefered skills' column very sensitively here. We cant just impute the Mean or high frequency values or something.  
# 1. I will try to find the null valued column's 'Salary range To' and find its similar salary's preferred skill and replace null. 
# 2. If no similar salary is found i will find range of +1000 and -1000 of 'Salary Range To' and then find a random preferred skill from the range of values found and replace the null value.
# 3. If found range of values are also having preferred values as null then skip the replace.

# In[18]:


replaced = False
for i, rec in enumerate(dFrame["Preferred Skills"]):
    if pd.isnull(rec):
        #print("For i", str(i), replaced)
        for j in range(0,dFrame.shape[0]):
            if i != j and dFrame["Salary Range To"].loc[i] == dFrame["Salary Range To"].loc[j]:
                #print(i, j, dFrame["Salary Range To"].loc[i], dFrame["Salary Range To"].loc[j] )
                if not dFrame["Preferred Skills"].loc[j] :
                    replaced  = True
                    #print(i, j, count, dFrame["Preferred Skills"].loc[j] )
                    dFrame["Preferred Skills"].loc[i] = dFrame["Preferred Skills"].loc[j]
                    break
        if not replaced :
            lis = dFrame.loc[(dFrame["Salary Range To"] > dFrame["Salary Range To"].loc[i]-1000) & (dFrame["Salary Range To"] < dFrame["Salary Range To"].loc[i]+1000)].index
            pickloc = random.choice(lis)
            clear = True
            con  = 0
            while (clear):
                if i != pickloc and not pd.isnull(dFrame["Preferred Skills"].loc[pickloc]):
                    dFrame["Preferred Skills"].loc[i] = dFrame["Preferred Skills"].loc[pickloc]
                    #print("From Picked",dFrame["Preferred Skills"].loc[pickloc])
                    replaced = False
                    clear = False
                pickloc = random.choice(lis)
                con += 1
                if con >20:
                    clear = False


# In[19]:


#dFrame.loc[(dFrame["Salary Range To"] > dFrame["Salary Range To"].loc[162]-1000) & (dFrame["Salary Range To"] < dFrame["Salary Range To"].loc[162]+1000)]
dFrame["Preferred Skills"].isnull().sum()


# In[20]:


# Now we have 3 records left to be imputed.
for i, rec in enumerate(dFrame["Preferred Skills"]):
    if pd.isnull(rec):
        print(i,str(dFrame["Salary Range To"][i]))


# In[21]:


dFrame.loc[976] 
# Belongs to Job Category = Engineering, Architecture, & Planning


# In[22]:


# Now let us know what this Job Category has more frequency of Skills
dg = dFrame.groupby(["Job Category"])
dgf = pd.DataFrame(dg.get_group("Engineering, Architecture, & Planning"))
dgf["Preferred Skills"].value_counts()


# In[23]:


# Impute the null value to the skill with more frequency under job category=Engineering, Arch...
dFrame["Preferred Skills"].loc[976] = "Ability to communicate effectively in verbal and written form.  Possession of a Motor Vehicle Driver's license valid in the State of New York is preferred.  Ability to work off-hours when necessary."


# In[24]:


# Lets look at the other 2 records.
dFrame.loc[1868:1869]


# Both are under Job Category= 'health'. So lets examin Health and find best fit to impute

# In[25]:


dg = dFrame.groupby(["Job Category"])
dgf = pd.DataFrame(dg.get_group("Health"))
dgf["Preferred Skills"].value_counts()


# In[26]:


#Lets impute the 'Preferred Skill' with highest frequency skill
dFrame["Preferred Skills"].loc[1868:1869] ='Knowledge of DOHMH and DOE personnel policies and procedures; excellent interpersonal, communication and presentation skills.'


# Imputing for "Minimum Qual Requirements" & "To Apply" and looking through "Posting Date","Posting Updated", "Process Date"

# Before imputing we shall understand the data and the info gain it provides for the current context of analysis (Skill/Salary/Job Category)

# In[27]:


dFrame[["To Apply", "Posting Date","Posting Updated", "Process Date"]]

# from this we can see that these columns are showing data which are of no use for any analysis, as the data is not 
#helpfull and unique. Hence we will remove these columns.


# In[28]:


dFrame = dFrame.drop(["To Apply", "Posting Date","Posting Updated", "Process Date"], axis=1)
#Remove the columns which are identified as not inferential


# Post removal of the columns as listed above we are left with few columns to analyse.

# In[29]:


print("Column count: ", str(len(dFrame.columns)))
dFrame.columns


# Lets examine some more columns which are not specific to job.

# In[30]:


#Columns which doesnt gove any info about the job or which doesn't adds up any info for statistical inference.
dFrame[['Job ID','# Of Positions','Title Code No','Work Location','Residency Requirement']]


# In[31]:


dFrame = dFrame.drop(['Job ID','# Of Positions','Title Code No','Work Location','Residency Requirement'], axis=1)
#Remove the columns identified as not inferential
print("Number of Columns left remaining post removal",str(len(dFrame.columns)))


# Lets look at the remaining columns

# In[32]:


dFrame.columns


# ## Analysis

# ### 1. What are the highest paid Skills in the US market?

# We shall sort the data in decending order in order to find the top payed salaries 1st

# In[33]:


sortedData = dFrame.sort_values('Salary Range To',ascending=False)


# In[34]:


sortedData.head(5)
# Top 5 Jobs with highest salary.


# In[35]:


# here we are looking for top 5 salaries and there skills as individually. 
# We can change the number of salaries to be considered.

top = 5
sentences = []
for rec in sortedData["Preferred Skills"].head(top):
    for sent in nltk.tokenize.sent_tokenize(rec):
        sentences.append(sent)
sentences = list(dict.fromkeys(sentences))


# In[36]:


# Considering each sentence in a paragraph of Preferred skill will be representing a skill. 
# We do sentence tokenizing to separate out each skill from the paragraph of Preferred skills.

print("These are Skill individually or in combination, highest paid skills in US Market")
for i, sen in enumerate(sentences):
    print("Skill ",i," = ",sen)


# ### 2. What are the job categories, which involve above mentioned niche skills?

# In[37]:


# Fetching the job category for the above identified Skills.
jCategories = []
for skill in sentences:
    for i, skillset in enumerate(sortedData["Preferred Skills"]):
        if skill in skillset:
            jCategories.append(sortedData["Job Category"][i])


# In[38]:


jCategories = list(dict.fromkeys(jCategories)) # Geting unique categories out of selected Job Categories
print("A total of ", str(len(jCategories)), " Job categories are identified to be under the identified top skills")
print("Categories are as below: ")
jCategories


# ### 3. Applying clustering concepts, please depict visually what are the different salary ranges based on job category and years of experience

# Preprocessing the 'Minimum Qual Requirements' column to find the Years of experience asked for the job.
# 1. For years not asked - 0 Years
# 2. Years as given 
# 3. For null values also - 0 Years

# In[39]:


# Creating a dictonary for the replacement of characters to numbers
nDic = {'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
        'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'thirteen':13, 'fourteen':14, 'fifteen':15, 
          'sixteen':16, 'seventeen':17, 'eighteen':18, 'nineteen':19,'twenty':20}    


# In[40]:


for j, qualification in enumerate(sortedData["Minimum Qual Requirements"]):
    if not pd.isnull(qualification):
        words = nltk.tokenize.word_tokenize(qualification)
        for i, word in enumerate(words):
            if word.lower() == "year" or word.lower() == "years":
                if not words[i-1].isnumeric() :
                    sortedData["Minimum Qual Requirements"][j] =str(nDic.get(words[i-1].lower()) if not pd.isnull(nDic.get(words[i-1].lower())) else '0') + " Year"
                    break
                elif words[i-1]==")":
                    sortedData["Minimum Qual Requirements"][j] = str(words[i-2]) + " Year"
                    break
                else:
                    sortedData["Minimum Qual Requirements"][j] = str(words[i-1]) + " Year"
                    break
    else:
        sortedData["Minimum Qual Requirements"][j] = "0 Year"


# Preprocessing "Job Category and "Minimum Qual Requirements" Features. 
# Label encoding to convert them to numerical values as K-Means takes only Numericals

# In[41]:


from sklearn.preprocessing import LabelEncoder
sortedData["Job Category"] = LabelEncoder().fit_transform(sortedData["Job Category"])
sortedData["Minimum Qual Requirements"] = LabelEncoder().fit_transform(sortedData["Minimum Qual Requirements"])


# Trying to find the optimal cluster number for the cluster algorithm K-Means

# In[42]:



from scipy.spatial.distance import cdist
clusteringData = sortedData[["Job Category","Minimum Qual Requirements"]]
cluster = []
score = []
for nCluster in range(2,30, 2):
    KMtest = KMeans(n_clusters=nCluster, random_state=0).fit(clusteringData)
    cluster.append(nCluster)
    score.append(KMtest.score(clusteringData))
    #score.append(sum(np.min(cdist(clusteringData, KMtest.cluster_centers_, 'euclidean'), axis=1)) / clusteringData.shape[0])

plt.plot(cluster, score, 'bx-')
plt.xlabel('Cluster')
plt.ylabel('Score')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# Optimal Cluster 11

# In[43]:


# From the elbow Curve we can identify that 11 cluster is the optimal cluster to use.
KMTrain = KMeans(n_clusters=11, random_state=0).fit(clusteringData)


# In[44]:


# We are creating the lists of Min and Max salaries identified for each cluster
minSal= []
maxSal = []
xClu = []
for cluster in range(0,11):
    xClu.append("Cluster"+str(cluster))
    minSal.append(sortedData.iloc[list(clusteringData[KMTrain.labels_== cluster].index),:]["Salary Range From"].min())
    maxSal.append(sortedData.iloc[list(clusteringData[KMTrain.labels_== cluster].index),:]["Salary Range To"].max())


# In[45]:


# Plotting the Salary ranges for each Cluster
ind = range(1,12)
fig = plt.figure(figsize=(15,8))
plt.bar(ind, maxSal, width=0.3,color='r')
plt.bar(ind, minSal, width=0.3,color='w')
plt.xticks(ind, xClu)
plt.ylabel('Salaries')
plt.xlabel('Clusters')
plt.title('Salary ranges based on Cluster from "Job Category and Year of Exp"')
plt.yticks(np.arange(0, 300000, 30000))
for i in range(len(ind)-1):
    plt.text(x = ind[i]-0.5 , y = minSal[i]-0.1, s = "Min:"+str(minSal[i]), size = 10)
    plt.text(x = ind[i]-0.5 , y = maxSal[i]+0.1, s = "Max:"+str(maxSal[i]), size = 10)
plt.show()

