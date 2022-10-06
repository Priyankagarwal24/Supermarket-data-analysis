#!/usr/bin/env python
# coding: utf-8

# In[163]:


# Importing the necessary libraries to import the data file in the code

# In[56]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('notebook')
sns.set_style('darkgrid')
sns.set_palette('Blues_r')


# In[57]:


dataset = pd.read_csv('supermarket_sales.csv')


# In[80]:


nRow, nCol = dataset.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[58]:


dataset


# Removing white space from the column names

# In[59]:


dataset.columns = dataset.columns.str.replace(' ', '')


# In[60]:


dataset.info()


# Checking whether any of the data entry is null or not

# In[61]:


dataset.isnull().sum().sort_values()


# In[62]:


dataset.columns


# Decribing the data in tabular as well as graphical format

# In[63]:


dataset.describe()


# In[64]:


dataset_to_plot = dataset.drop(columns=['InvoiceID', 'Branch', 'City', 'Customertype', 'Gender',
       'Productline', 'Date','Time', 'Payment','Rating','grossmarginpercentage']).select_dtypes(include=np.number)

dataset_to_plot.plot(subplots=True, layout=(3,3), kind='box', figsize=(15,12), patch_artist=True)
plt.subplots_adjust(wspace=0.5);


# Converting the date column to date-time data type to make it accessible in real time situations

# In[65]:


dataset['Date'] = pd.to_datetime(dataset.Date)


# In[66]:


dataset.Date


# In[67]:


dataset.rename(columns = {'Payment':'Paymentmethod'}, inplace = True)
dataset['CustomerMonth '] = pd.DatetimeIndex(dataset['Date']).month
dataset.head(2)


# Finding number of unique entries in each column

# In[68]:


dataset.nunique()


# Finding the correlation between different columns of the data set.
# Since gross margin is constant throughout the data set, it is not correlated with any of the columns.

# In[69]:


dataset.corr()


# In[70]:


#correlation map
f,ax = plt.subplots(figsize=(12, 12))
sns.heatmap(dataset.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[118]:


plt.figure(figsize=(5,4))
dataset.groupby('City')['Quantity'].sum().sort_values(ascending=False).plot(kind='bar')
plt.title('Total Number of Purchases by City', size=16)
plt.ylabel('Number of Purchases');
plt.xticks(rotation=0)


# In[119]:


plt.figure(figsize=(5,4))
dataset.groupby('Branch')['Total'].sum().plot(kind='bar')
plt.title('Total Amount spent by  branch', size=16)
plt.ylabel('Amount spent');
plt.xticks(rotation=0)


# In[120]:


plt.figure(figsize=(5,4))
dataset.groupby('Branch')['grossincome'].sum().plot(kind='bar')
plt.title('Gross Income by  branch', size=16)
plt.ylabel('Gross Income');
plt.xticks(rotation=0)


# In[121]:


plt.figure(figsize=(5,4))
dataset.groupby('Branch')['Rating'].sum().plot(kind='bar')
plt.title('Best Rating', size=16)
plt.ylabel('Rating');
plt.xticks(rotation=0)


# In[158]:


plt.figure(figsize=(8,3))
sns.displot(x=dataset['Rating'], kde=False, bins=12);
plt.title('Rating', size=16)
plt.ylabel('count');


# Checking the demand of different product types to analyze the inventory required

# In[109]:


sns.countplot(y = 'Productline', data = dataset, palette = "mako");


# Decribing branch-wise statistics

# In[110]:


B=dataset.groupby('Branch')


# In[111]:


for branch,data in B:
    print('branch name :', branch)
    print('--------------------------')
    print('data:',data)
    #print(data.describe())


# Checking the business statistics of Branch C to predict its behaviour

# In[112]:


x=B.get_group('C')
x


# In[92]:


branch_C=pd.DataFrame(x)


# In[93]:


branch_C.describe()


# In[96]:


Customer_type=branch_C.groupby('Customertype')[['grossincome']].mean().reset_index()


# In[97]:


Customer_type


# In[100]:


sns.catplot(x='Customertype',kind='count',data=branch_C);


# On an average members are buying from the store more often, but the ratio can be increased 

# In[129]:


dataset['weekday']=data['Date'].dt.day_name()


# In[137]:


w= dataset.groupby('weekday')[['grossincome']].sum().reset_index()
w.sort_values('weekday',ascending=True)
w


# In[138]:


sns.barplot(x='weekday',y='grossincome',data=w)
plt.xticks(rotation=45);


# The sales on saturday are higher as its weekend but strangely the sales are very good on weekdays "Tuesday" and "Wednesday".

# In[142]:


sns.catplot(x='Paymentmethod',kind='count',data=branch_C);


# In the generation of e-wallet payments, the amount of business done using Cash mode is pretty high. This is some strange behaviour.

# In[145]:


sns.catplot(x='Gender',hue='Customertype',kind='count',data=branch_C);


# In[147]:


Products=branch_C.groupby('Productline')[['grossincome']].mean().reset_index()
Products


# In[152]:


sns.catplot(x='Productline',kind='count',data=branch_C);
plt.xticks(rotation = 90);

