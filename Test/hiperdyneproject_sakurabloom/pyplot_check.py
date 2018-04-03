import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import  datetime

d = []
k = range(0,5)
blossom_date = datetime.date(1960, 1, 31)
blossom_day = datetime.timedelta(days=1)
for i in k:
    blossom_date = blossom_date + blossom_day
    s = blossom_date.strftime("%B %d")
    d.append(s)

print(d)

value_list = np.random.randint(0, 99, size = len(d))
pos_list = np.arange(len(d))

ax = plt.axes()
ax.xaxis.set_major_locator(ticker.FixedLocator((pos_list)))
ax.xaxis.set_major_formatter(ticker.FixedFormatter((d)))
plt.bar(pos_list, value_list, color = '.75', align = 'center')

plt.show()

from sklearn import metrics
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import ensemble
import datetime


sakura_df = pd.read_csv('sakura_withbloom.csv')
#print(sakura_df)

train_df = sakura_df
test_years = np.array([1966, 1971, 1985, 1994, 2008])
for i in test_years:
    train_df = train_df[train_df['year'] != i]

test_df = sakura_df
test_df = test_df[(test_df['year'] == 1966) | (test_df['year'] == 1971) | (test_df['year'] == 1985) | (test_df['year'] == 1994) | (test_df['year'] == 2008)]

accumulated_temp = []
tmax = 0
train_years = train_df.year.unique()
test_years = test_df.year.unique()
sakura_years = sakura_df.year.unique()

dts = 0
Ea = 5
DTSj = []
count = 0
R = 0.0019870937
Ts = 17

for i in train_df[train_df.year == 1961].serial:
    Tij = train_df.get_value(i + 39 - 1, 'avg temp')
    print(Tij)
    dts = dts + np.exp((Ea * (Tij - Ts)) / (R * Tij * Ts))
    count = count + 1
    if (count == (91 - 39 + 1)):
        break

DTSj.append(dts)
dts = 0
count = 0
