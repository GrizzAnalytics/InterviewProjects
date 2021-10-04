#!/usr/bin/env python
# coding: utf-8

# ## Standardized Test Score File Processing

# In[1]:


# importing packages

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# loading both csvs
pd = pd.read_csv('sample-pd.csv')
pdProcessed = pd.read_csv('sample-pd-processed.csv')


# In[3]:


print(pd.shape)
pd.head()


# In[4]:


pd.info(max_cols = 238)


# In[5]:


pd.describe()


# In[6]:


pd.nunique()


# In[7]:


pdProcessed.head()


# In[8]:


pdProcessed.info()


# In[9]:


pdProcessed.describe()


# In[10]:


pdProcessed.nunique()


# Now I can start working on showing the `pd` dataset cleaned up to specifications. Once I do that, I can  If need be, I can give the cleaned data set one more quick programatic assessment and cleaning if needed.

# In[11]:


# creating a new df for sample-pd.csv
pd_clean = pd.copy()
print(pd_clean.shape)
pd_clean.head(5)


# In[12]:


pd_clean.info(max_cols = 238)


# In[13]:


# dropping unnecessary columns
pd_clean = pd_clean.drop(["some_columns"], axis = 1)


# In[14]:


pd_clean.info()


# In[15]:


pd_processed = pd_clean.copy()
pd_processed.head()


# In[16]:


#adding LocalID to match pdP
pd_processed['LocalID'] = " "
pd_processed['NameType'] = " "

# adding NCESID to match to pdP
pd_processed['NCESID'] = 373737
pd_processed['NCESID'] = pd_processed['NCESID'].astype(int)

# adding TestName to match pdP
pd_processed['TestName'] = 'pd'

# adding GradeLevel to match pdP
pd_processed['GradeLevel'] = pd_processed['stugrade']

print(pd_processed.info())
pd_processed.head()


# In[17]:


# renaming sasid to StudentTestId to match pdP
pd_processed.rename(columns = {'sasid' : 'StudentTestID'}, inplace = True)

# renaming stugrade to StudentGradeLevel
pd_processed.rename(columns = {'stugrade' : 'StudentGradeLevel'}, inplace = True)

pd_processed.head()


# In[18]:


pd_processed = pd.concat([pd_processed] * 3, ignore_index = True)
pd_processed.info()


# In[19]:


pd_processed = pd_processed.sort_values('StudentTestID')


# In[20]:


# creating NameType column
vals1 = (['April 1', 'May 1', 'June 1'] * 2671)
pd_processed['TestDate'] = pd.DataFrame({'TestDate' : vals1})

# creating NameType column
vals2 = (['pd ELA', 'pd Math', 'pd Science'] * 2671)
pd_processed['NameType'] = pd.DataFrame({'NameType' : vals2})

# creating NameType column
vals3 = (['ELA', 'Math', 'Science'] * 2671)
pd_processed['TestSubjectName'] = pd.DataFrame({'TestSubjectName' : vals3})


# In[21]:


pd_processed['NameType'].value_counts()


# In[22]:


pd_processed['TestSubjectName'].value_counts()


# In[23]:


pd_processed['TestDate'].value_counts()


# In[24]:


pd_processed = pd_processed.sort_values('StudentTestID')
print(pd_processed.shape)
pd_processed.head(9)


# In[25]:


pd_processed.info()


# In[26]:


# defining performance functions

def performance(row):
    label = pd_processed['Score1Label']
    for label in row['Score1Label']:
        if row['NameType'] == 'pd ELA':
            return row['eperf2']
        elif row['NameType'] == 'pd Math':
            return row['mperf2']
        elif row['NameType'] == 'pd Science':
            return row['sperf2']
        return

# creating Score1 Columns (type, label, and score columns)
pd_processed['Score1Label'] = 'Performance Level'
pd_processed['Score1Type'] = 'Level'
pd_processed['Score1Value'] = pd_processed.apply(performance, axis = 1)


# In[27]:


# defining scaled function

def scaled(row):
    label = pd_processed['Score2Label']
    for label in row['Score2Label']:
        if row['NameType'] == 'pd ELA':
            return row['escaleds']
        elif row['NameType'] == 'pd Math':
            return row['mscaleds']
        elif row['NameType'] == 'pd Science':
            return row['sscaleds']
        return

# creating Score2 Columns (type, label, and score columns)
pd_processed['Score2Label'] = 'Scaled Score'
pd_processed['Score2Type'] = 'Scale'
pd_processed['Score2Value'] = pd_processed.apply(scaled, axis = 1)


# In[28]:


# defining cpi function

def cpi(row):
    label = pd_processed['Score3Label']
    for label in row['Score3Label']:
        if row['NameType'] == 'pd ELA':
            return row['ecpi']
        elif row['NameType'] == 'pd Math':
            return row['mcpi']
        elif row['NameType'] == 'pd Science':
            return row['scpi']
        return
    
# creating Score3 Columns (type, label, and score columns)
pd_processed['Score3Label'] = 'CPI'
pd_processed['Score3Type'] = 'Scale'
pd_processed['Score3Value'] = pd_processed.apply(cpi, axis = 1)


# In[29]:


# creating Score4 Columns (type, label, and score columns)
pd_processed['Score4Label'] = np.nan
pd_processed['Score4Type'] = np.nan
pd_processed['Score4Value'] = np.nan


# In[30]:


pd_processed.head(3)


# In[31]:


pd_processed = pd_processed.drop(columns = ['escaleds', 'eperf2', 'ecpi',
                                                'mscaleds', 'mperf2', 'mcpi',
                                                'sscaleds', 'sperf2', 'scpi'])


# In[32]:


pd_processed.info()


# In[33]:


pd_processed.head(9)


# In[34]:


pd_processed['Score1Value'].value_counts()


# In[35]:


pd_processed['Score1Value'].value_counts()


# In[36]:


#remapping Score1Value values to match pdProcessed

pd_processed.Score1Value = pd_processed.Score1Value.replace('F', '1-P')
pd_processed.Score1Value = pd_processed.Score1Value.replace('W', '2-W') 
pd_processed.Score1Value = pd_processed.Score1Value.replace('NI', '3-NI') 
pd_processed.Score1Value = pd_processed.Score1Value.replace('P', '4-P') 
pd_processed.Score1Value = pd_processed.Score1Value.replace('A', '5-A') 
pd_processed.Score1Value = pd_processed.Score1Value.replace(' ', 'P+') 
pd_processed.Score1Value = pd_processed.Score1Value.replace('P+', '6-P+') 


# In[37]:


# rearraging the order of columns
pd_processed = pd_processed[['NCESID', 'StudentTestID', 'LocalID', 'StudentGradeLevel',
                                 'TestDate', 'TestName', 'NameType', 'TestSubjectName',
                                 'GradeLevel', 'Score1Label', 'Score1Type', 'Score1Value',
                                 'Score2Label', 'Score2Type', 'Score2Value', 'Score3Label',
                                 'Score3Type', 'Score3Value', 'Score4Label', 'Score4Type',
                                 'Score4Value']]


# In[38]:


# converting datatypes
pd_processed['NCESID'] = pd.to_numeric(pd_processed['NCESID'], errors = 'coerce')
pd_processed['StudentGradeLevel'] = pd.to_numeric(pd_processed['StudentGradeLevel'], errors = 'coerce')
pd_processed['GradeLevel'] = pd.to_numeric(pd_processed['GradeLevel'], errors = 'coerce')
pd_processed['Score2Value'] = pd.to_numeric(pd_processed['Score2Value'], errors = 'coerce')
pd_processed['Score3Value'] = pd.to_numeric(pd_processed['Score3Value'], errors = 'coerce')


# In[39]:


pd_processed.head(9)


# In[40]:


pd_processed.info()


# In[41]:


pd_processed.to_csv('sample-pd-processed-new.csv', index = False, encoding = 'utf-8')


# ## Conclusion

# I wanted to showcase my python skills (as that has been my main language focus for a significant portion of my bootcamp, and personal projects. Although I am not entirely fluent in the pandas library, I went with a route that would best suit my strengths. I believe to run this for 20 files, I would need to only alter the initial csv that the data is being pulled from as well as the title of the stored output csv. However, I still have so much to learn, and logically it means I could be more off than I think I am. With that said, I am very confident in my current abilities and was able to finish this project, run it, and compare the `sample-pd-processed.csv` and the `sample-pd-processed-new.csv` structure successfully.
# 
# I've also noticed that there were some tidiness issues regarding the datatypes of the cleaned and manipulated dataframe. Namely, the 'StudentGradeLevel' and 'GradeLevel' datatypes could not be converted to integers, as there were three values that equalled 'SP', which is a string (yet is being read as a float). Also there is a fourth score in `sample-pd-processed` but there are only three score types.
# 
# Lastly, I wasn't sure if you were interested in seeing some visualizations, so instead, I'll include them after this. This way, I can still showcase my skill without taking away from the assigment itself.

# ### General Visualization

# In[42]:


# setting up the seaborn visualization style
sb.set(style = "white", context = "notebook")


# In[43]:


# creating separate dataframes to visualize better
ela_scores = pd_processed[pd_processed['TestSubjectName'] == 'ELA']
math_scores = pd_processed[pd_processed['TestSubjectName'] == 'Math']
sci_scores = pd_processed[pd_processed['TestSubjectName'] == 'Science']


# In[44]:


# creating pie chart
plt.title('ELA Performance', size = 20)
ela_scores['Score1Value'].value_counts().plot(kind = 'pie', figsize = (8,8), fontsize = 15)
plt.ylabel('');


# In[45]:


plt.title('Math Performance', size = 20)
math_scores['Score1Value'].value_counts().plot(kind = 'pie', figsize = (8,8), fontsize = 15)
plt.ylabel('');


# In[46]:


plt.title('Science Performance', size = 20)
sci_scores['Score1Value'].value_counts().plot(kind = 'pie', figsize = (8,8), fontsize = 15)
plt.ylabel('');


# In[47]:


pd_processed['Score2Value'].hist(figsize = (10,10));


# In[48]:


math_scores['Score2Value'].hist(figsize = (10,10));


# In[49]:


sci_scores['Score2Value'].hist(figsize = (10,10));

