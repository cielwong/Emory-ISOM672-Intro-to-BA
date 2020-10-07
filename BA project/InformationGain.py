import pandas as pd
import numpy as np
import os
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

os.chdir('C:\\Users\\dongh\\Downloads')
df = pd.read_csv('C:\\Users\\dongh\\Downloads\\hotel_bookings.csv')

#print(df)

print("shape", df.shape)
print("dtypes", df.dtypes)

df_y = df.iloc[:,1]
df_x = df.iloc[:, 2:32]
print(df_x.describe())

print(df_x.columns)

df2 = df #making a copy of original dataframe
df_cat = df2[['hotel','arrival_date_month', 'meal', 'country', 'market_segment', 
          'distribution_channel', 'reserved_room_type','assigned_room_type',
          'deposit_type', 'customer_type', 'reservation_status']]
df_cat.shape
for i in range(0,11):
        print("value counts", df_cat.iloc[:,i].value_counts())

#df.replace('Undefined', np.NaN)

######################## Recoding non-numeric categorical variables
# pd.factorize?

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# drop country for now
df2 = df # making a copy of df
df_cat = df2[['hotel','arrival_date_month', 'meal', 'country', 'market_segment', 
          'distribution_channel', 'reserved_room_type','assigned_room_type',
          'deposit_type', 'customer_type', 'reservation_status']]
cleanup_categ = {"hotel":     {"Resort Hotel": 0, "City Hotel": 1},
                "arrival_date_month": {"January": 1, "February": 2, "March": 3, "April": 4,
                                       "May": 5, "June": 6, "July": 7, "August": 8, "September": 9, 
                                       "October": 10, "November": 11, "December": 12},
                "meal": {"Undefined": 0, "SC":1, "BB":2, "HB":3, "FB":4 },
                "market_segment": {"Online TA": 0, "Offline TA/TO": 1, "Groups":2, "Direct":3, "Corporate": 4, 
                                   "Complementary":5, "Aviation":6, "Undefined": 7},
                "distribution_channel": {"TA/TO": 0, "Direct": 1, "Corporate":2, "GDS":3, "Undefined":4},
                "reserved_room_type": {"A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":6, "H":7, "L":8, "P":9},
                "assigned_room_type": {"A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":6, "H":7, "I":8, "K":9, "L":10, "P":11},
                "deposit_type": {"No Deposit":0, "Non Refund":1, "Refundable":2},
                "customer_type": {"Contract": 0, "Group":1, "Transient":2, "Transient-Party":3},
                "reservation_status": {"Canceled":0, "Check-Out":1, "No-Show":2}}

df2.replace(cleanup_categ, inplace = True)

df2 = df2.drop(['is_canceled','children', 'agent', 'company', 'adr'], axis=1)
df2.shape
df2.head(5)
######################################## GET INFORMATION GAIN ####################################

from sklearn.feature_selection import mutual_info_classif

info_gain = {}
# attributes excludes country, reservation date for now
attributes = ['hotel','lead_time', 'arrival_date_year', 'arrival_date_month',
       'arrival_date_week_number', 'arrival_date_day_of_month',
       'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children',
       'babies', 'meal', 'market_segment', 'distribution_channel',
       'is_repeated_guest', 'previous_cancellations',
       'previous_bookings_not_canceled', 'reserved_room_type',
       'assigned_room_type', 'booking_changes', 'deposit_type', 'agent',
       'company', 'days_in_waiting_list', 'customer_type', 'adr',
       'required_car_parking_spaces', 'total_of_special_requests',
       'reservation_status']
for i in range(0,24):
        x = df2.iloc[:, i] # subsets a column
        x = pd.DataFrame(x)
        ig = mutual_info_classif(x, df_y, discrete_features=False)
        ig = ig.tolist() # convert array to list
        ig = str(ig).strip('[]') # convert list to string value
        ig = float(ig)
        info_gain[attributes[i]] = ig # saves information gain in info_gain dictionary

sort_ig = sorted(info_gain.items(), key=lambda x: x[1], reverse=True)

for each in sort_ig:
    print(each[0], each[1])
