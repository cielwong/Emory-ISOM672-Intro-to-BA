
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype, is_numeric_dtype
import numpy as np
from scipy.stats import entropy, ttest_ind
from sklearn.preprocessing import LabelEncoder

#%%
data = pd.read_csv("C:/Users/10331/OneDrive/Documents/GitHub/Emory-ISOM672-Intro-to-BA/BA project/hotel_bookings.csv")
data.shape
data.describe()
data.dtypes
data.head()

is_string_dtype(data["hotel"])

data.isnull().sum()
data.groupby("arrival_date_year").agg(["mean","std"])

#%% Feature Selection 1
df = data[:]
df["country"] = df["country"].fillna("no_fill")

#df = df.dropna(subset=["country"])
df["children"] = df["children"].fillna(0)
df["agent"] = df["agent"].fillna("no_agent")
df.isnull().sum()

#df.drop("arrival_date_year",axis =1, inplace = True)
#df.drop("arrival_date_day_of_month",axis =1, inplace = True)

df.drop("reservation_status",axis =1, inplace = True)
df.drop("company",axis =1, inplace = True)
df.drop("assigned_room_type",axis =1, inplace = True)
df.drop("reservation_status_date",axis =1, inplace = True)

df["is_canceled"] = df["is_canceled"].astype(str)
df["agent"] = df["agent"].astype(str)
df["is_repeated_guest"] = df["is_repeated_guest"].astype(str)

df.dtypes
df.value_counts()

tmp = []
for i in df.columns:
    tmp.append(df[i].nunique())
    
#Unique count
col = list(df.columns)
unique_count = pd.DataFrame(list(zip(col, tmp)),columns = ["Columns","Unique_count"])
unique_count

#Histogram for numeric
for i in range(len(df. columns)):
    if is_numeric_dtype(df.iloc[:,i]):
        sns.distplot(df.iloc[:,i],kde=False)
        plt.show()
        if sum(df.iloc[:,i]>(df.iloc[:,i].quantile(0.75)*2.5 - df.iloc[:,i].quantile(0.25)*1.5)) > 0:
            sns.distplot(df.iloc[:,i],kde=False)
            plt.ylim(0,10)
            plt.xlim(df.iloc[:,i].quantile(0.75),df.iloc[:,i].max()+5)
            plt.axvline(2.8, 0, df.iloc[:,i].quantile(0.75)*2.5 - df.iloc[:,i].quantile(0.25)*1.5, color = "red")
            plt.show()

#country need regroup by regions (except the "big ones")
#https://github.com/lukes/ISO-3166-Countries-with-Regional-Codes/blob/master/all/all.csv


# lead_time transform
df.lead_time = (df["lead_time"])**0.5
# np.log()
sns.distplot(df.lead_time)

#%% t-test for cancellation

c1 = df.iloc[:,1][df["arrival_date_year"] == 2015]
c2 = df.iloc[:,1][df["arrival_date_year"] == 2016]
c3 = df.iloc[:,1][df["arrival_date_year"] == 2017]

ttest_ind(c1, c2, equal_var = False)
ttest_ind(c1, c3, equal_var = False)
ttest_ind(c2, c3, equal_var = False)

#%% information gain

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


df_y = df.iloc[:,1].to_frame()
df_x = df[:]
df_x.drop("is_canceled",axis = 1)

df_cat = df_x.loc[:,['hotel','arrival_date_month', 'meal','market_segment','distribution_channel',
                     'reserved_room_type','deposit_type', 'customer_type','country']]
x2 = df_x[["is_repeated_guest","agent", "arrival_date_week_number","arrival_date_year"]]
x2 = x2.fillna(0)
x2.isnull().sum()

x1 = df_cat.apply(LabelEncoder().fit_transform)
x = pd.concat([x1,x2],axis = 1)
 
col = list(x.columns)
ig = list(mutual_info_classif(x, df_y, discrete_features=True))

info_gain = pd.DataFrame(list(zip(col, ig)),columns = ["Columns","Info_gain"]).sort_values(by=['Info_gain'],ascending=False)
info_gain

#%% Heatmap CM

df2 = data.iloc[:, [1,2,7,8,9,10,11,17,18,21,25,27,28,29]]
corr= df2.corr()
mask = np.triu(np.ones_like(corr))

plt.figure(figsize=(16, 16))
ax = sns.heatmap(
    corr,
    vmin=-.5, vmax=.5, center=0,
    cmap= sns.diverging_palette(20,220, n=200),
    square = True, 
    linewidth=4,
    annot = True, 
    fmt='.1g',
    mask = mask

    )

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment = 'right'
    )

#%% Histogram plots
for i in range(len(df. columns)):
    if is_string_dtype(df.iloc[:,i]):
        sns.countplot(df.iloc[:,i])
        plt.show
    else:
        sns.distplot(df.iloc[:,i],kde=False)
        plt.show()
        if sum(df.iloc[:,i]>(df.iloc[:,i].quantile(0.75)*2.5 - df.iloc[:,i].quantile(0.25)*1.5)) > 0:
            sns.distplot(df.iloc[:,i],kde=False)
            plt.ylim(0,10)
            plt.xlim(df.iloc[:,i].quantile(0.75),df.iloc[:,i].max()+5)
            plt.show()

#%% Scatter plots
num = list(df. columns)
for i in list(df. columns):
    for j in num:
        if is_string_dtype(df.loc[:,i]):
            sns.catplot(i,j,data = df)
            plt.show
        elif is_string_dtype(df.loc[:,j]):
            sns.catplot(j,i,data = df)
            plt.show
        else:
            sns.relplot(i,j,data = df)
            plt.show()
    del num[0]

#%% Scatter plots against Y
for i in list(df. columns):
    sns.catplot("is_canceled",i,data = df)
    plt.show()

#%% Agent working on 
tmp1 = data.loc[:,["agent","arrival_date_year"]]
tmp1["agent"] = tmp1["agent"].fillna(0)
tmp1 = tmp1.groupby(["agent","arrival_date_year"]).size()
tmp1 = tmp1.reset_index()
tmp1.dtypes

# >100 in 2017
tmp = tmp1[tmp1.arrival_date_year == 2017]
tmp.groupby(0).size()
tmp = tmp[tmp[0] > 100]
tmp1 = tmp1[tmp1.agent.isin(tmp.agent)]

# >500
tmp = tmp1.groupby(["agent"]).sum()
tmp = tmp[tmp[0] > 500]

tmp1 = data.loc[:,"agent"].to_frame()
tmp1["agent"] = tmp1["agent"].fillna(0)
tmp = tmp1[tmp1.isin(list(tmp.index))]

tmp.value_counts()
tmp.isnull().sum()
tmp["agent"] = tmp["agent"].fillna(0.1)

df.agent = tmp

tmp.value_counts()

#%% Children Babies adults

tmp = df.loc[:,["is_canceled","babies"]]
tmp.groupby(["babies","is_canceled"]).size()
df = df[df.babies != 9]
df = df[df.babies != 10]
df.loc[df["babies"]==2,"babies"] = 1

tmp = df.loc[:,["is_canceled","children"]]
tmp.groupby(["children","is_canceled"]).size()
df = df[df.children != 10]
df.loc[df["children"]==3,"children"] = 2

tmp = df.loc[:,["is_canceled","adults"]]
tmp.groupby(["adults","is_canceled"]).size()
df.drop(df.loc[df['adults'] > 4].index, inplace=True)
df.loc[df["adults"]==4,"adults"] = 3

#%% stays_in_weekend_nights

tmp = df.loc[:,["is_canceled","stays_in_weekend_nights"]]
tmp.groupby(["stays_in_weekend_nights","is_canceled"]).size()
tmp = tmp.groupby(["stays_in_weekend_nights","is_canceled"]).size().reset_index(name='count')
a = tmp.groupby('stays_in_weekend_nights')['count'].transform('sum')
tmp['count'] = tmp['count'].div(a)
tmp

#%%

# Checking Percentages of is_canceled for each label
tmp = df.loc[:,["is_canceled","adults"]]
tmp.groupby(["adults","is_canceled"]).size()
tmp = tmp.groupby(["adults","is_canceled"]).size().reset_index(name='count')
a = tmp.groupby('adults')['count'].transform('sum')
tmp['count'] = tmp['count'].div(a)
tmp

#Checking the distribution of the column
tmp = df.loc[:,["lead_time"]]
tmp.groupby(["lead_time"]).size()/sum(tmp.count())

#%% Regions

cc=pd.read_csv("C:/Users/10331/OneDrive/Documents/GitHub/Emory-ISOM672-Intro-to-BA/BA project/Region_Label.csv")
cc.head()
country_dict = dict(zip(cc["alpha-3"], cc["sub-region"]))

tmp1 = df.loc[:,["country"]]
tmp1 = tmp1.value_counts().reset_index()
tmp1 = list(tmp1[tmp1[0] < 488]["country"])

for key in list(country_dict.keys()):
    if key in tmp1:
        print(1)
    else:
        print(0)
        del country_dict[key]
        
df.replace({"country": country_dict}, inplace = True)
df.head()
df["country"].value_counts()

tmp1 = df.loc[:,["country"]]
tmp1 = tmp1.value_counts().reset_index()
tmp1 = list(tmp1[tmp1[0] < 488]["country"])
tmp = df["country"].to_frame()
tmp[tmp.country.isin(tmp1)] = "others"
df["country"] = tmp

df["country"].value_counts()

#%% adr

tmp = df.loc[:,["arrival_date_year","arrival_date_month","lead_time","adr"]]
tmp.dtypes
tmp.head()

tmp1 = tmp["arrival_date_month"].to_frame()
cleanup_categ =  {"January": "1", "February": "2", "March": "3", "April": "4",
                 "May": "5", "June": "6", "July": "7", "August": "8", "September": "9", 
                 "October": "10", "November": "11", "December": "12"}
tmp1.replace(cleanup_categ, inplace = True)
tmp1.head(10)
tmp1["arrival_date_month"] = tmp1["arrival_date_month"].astype(int)
tmp.arrival_date_month = tmp1.arrival_date_month

tmp.lead_time = tmp["lead_time"]//7
month = tmp["lead_time"]//4
year = month//12
tmp.arrival_date_year = tmp.arrival_date_year - year
month = month%12
tmp.arrival_date_month = tmp.arrival_date_month - month

for i in range(len(tmp)):
    if tmp.iloc[i,1] < 1:
        tmp.iloc[i,1] = 12 + tmp.iloc[i,1]
        tmp.iloc[i,0] = tmp.iloc[i,0] - 1
        
tmp = tmp.loc[:,["arrival_date_year","arrival_date_month"]]
tmp.groupby(["arrival_date_year","arrival_date_month"]).size()

df.arrival_date_year = tmp.arrival_date_year
df.arrival_date_month = tmp.arrival_date_month

df = df[df["arrival_date_year"]!=2013]
for j in [1,3,4,4,4,5,6,6,6,7]:
    for i in range(len(df)):
        if df.iloc[i,3] == 2014:
            if df.iloc[i,4] == j:
                df = df.drop(df.index[i])
                break
df = df[df["adr"] != 0]
df = df[df["adr"]<500]

tmp = df.loc[:,["arrival_date_year","arrival_date_month","adr"]]
tmp1 = tmp.loc[:,["arrival_date_year","arrival_date_month"]]
tmp1.groupby(["arrival_date_year","arrival_date_month"]).size()

tmp1 = tmp.groupby(["arrival_date_year","arrival_date_month"]).mean().reset_index()
tmp1

tmp1.arrival_date_month = tmp1.arrival_date_month + 1
for i in range(len(tmp1)):
    if tmp1.iloc[i,1] == 13:
        tmp1.iloc[i,1] = 1
        tmp1.iloc[i,0] = tmp1.iloc[i,0] + 1
tmp1 = tmp1[:-1]

for j in [8,8,8,8,8,8,8,8,8,8]:
    for i in range(len(df)):
        if df.iloc[i,3] == 2014:
            if df.iloc[i,4] == j:
                df = df.drop(df.index[i])
                break
tmp = df.loc[:,["arrival_date_year","arrival_date_month","adr"]]

for i in [2014,2015,2016,2017]:
    for j in [1,2,3,4,5,6,7,8,9,10,11,12]:
        if (i == 2014)&(j < 9):
            print(0)
            continue
        elif (i == 2017) & (j > 8):
            print(1)
            continue
        avg = float(tmp1.loc[(tmp1["arrival_date_year"]==i)&(tmp1["arrival_date_month"]==j),"adr"])
        tmp.loc[(tmp["arrival_date_year"]==i)&(tmp["arrival_date_month"]==j),"adr"] = (tmp.loc[(tmp["arrival_date_year"]==i)&(tmp["arrival_date_month"]==j),"adr"].to_frame() - avg)/avg
        
tmp.isnull().sum()
tmp.arrival_date_year.value_counts()
tmp.arrival_date_month.value_counts()
tmp.adr.value_counts()

sns.distplot((tmp.adr)**0.33)
