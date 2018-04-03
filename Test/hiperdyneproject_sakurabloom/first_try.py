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

p = range(0,52)
for k in p:
    for i in train_df[train_df.year == train_years[k]].serial:
        max_temp = train_df.get_value(i+31, 'max temp')
        tmax = tmax + max_temp
        if(train_df.get_value(i+31, 'bloom') == 1):
            accumulated_temp.append(tmax)
            break
    tmax = 0
print(len(accumulated_temp))
print(len(train_years))

Tmean = np.mean(accumulated_temp)
print(Tmean)

sh = np.ones(len(train_years))*600
tm = np.ones(len(train_years))*Tmean

plt.plot(train_years, accumulated_temp)
plt.plot(train_years, sh)
plt.plot(train_years, tm)
plt.xlabel('train years')
plt.ylabel('accumulated max temperature')
plt.show()

p = range(0, 5)
tmax = 0
days = 31
days_tmean = []
for k in p:
    for i in test_df[test_df.year == test_years[k]].serial:
        max_temp = test_df.get_value(i + 31, 'max temp')
        tmax = tmax + max_temp
        days = days + 1
        if (tmax > Tmean):
            days_tmean.append(days)
            break
    tmax = 0
    days = 31

tmax = 0
days = 31
days_sixhundred = []
for k in p:
    for i in test_df[test_df.year == test_years[k]].serial:
        max_temp = test_df.get_value(i + 31, 'max temp')
        tmax = tmax + max_temp
        days = days + 1
        if (tmax > 600):
            days_sixhundred.append(days)
            break
    tmax = 0
    days = 31

days = 31
days_true = []
for k in p:
    for i in test_df[test_df.year == test_years[k]].serial:
        days = days + 1
        if (test_df.get_value(i + 31, 'bloom') == 1):
            days_true.append(days)
            break
    days = 31

print(days_tmean)
print(days_sixhundred)
print(days_true)

diff_tmean = np.subtract(days_true, days_tmean)
print(diff_tmean)

diff_sixhundred = np.subtract(days_true, days_sixhundred)
print(diff_sixhundred)

jan = np.ones(len(days_tmean))*31
days_tmean_feb = np.subtract(days_tmean, jan)

days_sixhundred_feb = np.subtract(days_sixhundred, jan)

days_true_feb = np.subtract(days_true, jan)


diff_zero = np.zeros(5)

from sklearn import metrics
r2_tmean = metrics.r2_score(days_true, days_tmean)
r2_sixhundred = metrics.r2_score(days_true, days_sixhundred)
r2_tmean_feb = metrics.r2_score(days_true_feb, days_tmean_feb)
r2_sixhundred_feb = metrics.r2_score(days_true_feb, days_sixhundred_feb)

print(r2_tmean)
print(r2_sixhundred)
print(r2_tmean_feb)
print(r2_sixhundred_feb)


phi = 35.6666666
l = 4
avg_temp = []
days = 0
accumulated_avg_temp = 0
p = range(0, 57)

for k in p:
    for i in sakura_df[sakura_df.year == sakura_years[k]].serial:
        if (sakura_df.get_value(i, 'month') == 4):
            break

        temp = sakura_df.get_value(i, 'avg temp')
        accumulated_avg_temp = accumulated_avg_temp + temp
        days = days + 1

    meantemp_pyear = accumulated_avg_temp/float(days)
    avg_temp.append(meantemp_pyear)
    accumulated_avg_temp = 0
    days = 0

bloom_days = []
for k in p:
    atemp = avg_temp[k]
    dj = 136.75 - 7.689 * phi + 0.133 * phi * phi - 1.307 * (np.log(l)) + 0.144 * atemp + 0.285 * atemp * atemp
    bloom_days.append(int(dj))

d = []
k = range(0,27,3)
blossom_date = datetime.date(1960, 1, 31)
blossom_day = datetime.timedelta(days=3)
for i in k:
    blossom_date = blossom_date + blossom_day
    s = blossom_date.strftime("%B %d")
    d.append(s)

y_lbl = np.arange(35.0, 57.0, 2.5)

plt.plot(sakura_years, bloom_days)
plt.xlabel('years')
plt.ylabel('hibernation end day')
plt.yticks(y_lbl, d)
plt.show()

R = 8.314
Ts = 17 + 273
dict_bloom = dict((k, i) for (k, i) in zip(sakura_years, bloom_days))
for i in test_years:
    del dict_bloom[i]

train_hiber_end_days = list(dict_bloom.values())
print(train_hiber_end_days)

p = range(0,52)
days = 31
train_bloom_days = []
for k in p:
    for i in train_df[train_df.year == train_years[k]].serial:
        days = days + 1
        if (train_df.get_value(i + 31, 'bloom') == 1):
            train_bloom_days.append(days)
            break;
    days = 31

print(train_bloom_days)

dts = 0
Ea = range(5, 41)
DTSj = []
count = 0

for r in Ea:
    for (m, j, k) in zip(p, train_bloom_days, train_hiber_end_days):
        for i in train_df[train_df.year == train_years[m]].serial:
            Tij = train_df.get_value(i + k - 1, 'avg temp') + 273
            dts = dts + np.exp((r * 4200 * (Tij - Ts)) / (R * Tij * Ts))
            count = count + 1
            if (count == (j - k + 1)):
                break

        DTSj.append(dts)
        dts = 0
        count = 0

DTSj1 = np.reshape(DTSj, (36, 52))
DTSj = np.reshape(DTSj, (36, 52)).T

DTSmean = []
for i in range(1, 52):
    plt.plot(Ea, DTSj[i], 'bo')

plt.plot(Ea, DTSj[0], 'bo', label='DTSj vs Ea')
for i in range(0, 36):
    DTSmean.append(np.mean(DTSj1[i]))

DTSabsmean = np.mean(DTSmean)
avgDTSmean = np.ones(len(Ea)) * DTSabsmean

plt.plot(Ea, DTSmean, 'yo', label='DTSmean for every Ea individually vs Ea')
plt.plot(Ea, avgDTSmean, 'r', label='DTSmean')
plt.xlabel('Activation Energy in kCal')
plt.ylabel('transformed temperature days, DTS')
plt.legend()
plt.show()

print(DTSmean)

mse = []
days_DTSmean = []
p = range(0,52)
dts = 0
count = 0

for r in Ea:
    for (m, j, k) in zip(p, train_bloom_days, train_hiber_end_days):
        for i in train_df[train_df.year == train_years[m]].serial:
            Tij = train_df.get_value(i + k - 1, 'avg temp') + 273
            dts = dts + np.exp((r * 4200 * (Tij - Ts)) / (R * Tij * Ts))
            count = count + 1
            if (dts > DTSmean[r-5]):
                break
            if((i + k -1) == 20543):
                break

        days_passed = k + count - 1
        days_DTSmean.append(days_passed)
        dts = 0
        #print(count)
        #print('\n')
        count = 0
    #print(r)
    mse_value = metrics.mean_squared_error(train_bloom_days, days_DTSmean)
    #print(days_DTSmean)
    mse.append(mse_value)
    del days_DTSmean[:]

plt.plot(Ea, mse, 'ro')
plt.xlabel('Activation Energy in kCal')
plt.ylabel('Mean Squared Error')
plt.show()

dict_mse = dict((k, i) for (k, i) in zip(Ea, mse))
best_Ea = min(dict_mse, key=dict_mse.get)
print(best_Ea)

dict_bloom = dict((k, i) for (k, i) in zip(sakura_years, bloom_days))
for i in train_years:
    del dict_bloom[i]

test_hiber_end_days = list(dict_bloom.values())
print(test_hiber_end_days)

days_DTSmean_test = []
p = range(0, 5)
dts = 0
count = 0
r2 = []

for r in Ea:
    for (m, k) in zip(p, test_hiber_end_days):
        for i in test_df[test_df.year == test_years[m]].serial:
            Tij = test_df.get_value(i + k - 1, 'avg temp') + 273
            dts = dts + np.exp((r * 4200 * (Tij - Ts)) / (R * Tij * Ts))
            count = count + 1
            if (dts > DTSmean[r-5]):
                break

        days_passed = k + count - 1
        days_DTSmean_test.append(days_passed)
        dts = 0
        count = 0
    r2_Ea = metrics.r2_score(days_true, days_DTSmean_test)
    r2.append(r2_Ea)
    del days_DTSmean_test[:]

#print(days_true)
#print(days_DTSmean_test)
#print(r2_best_Ea)
plt.plot(Ea, r2)
plt.show()
dict_r2 = dict((k, i) for (k, i) in zip(Ea, r2))

best_Ea_r2 = max(dict_r2, key=dict_r2.get)
print(best_Ea_r2)
max_r2 = max(dict_r2.values())
print(max_r2)

#############################
datanow = np.zeros(11)
train_data = []
p = range(0, 51)
count = 0

for m in p:
    for i in train_df[train_df.year == train_years[m]].serial:
        datanow[0] = datanow[0] + train_df.get_value(i, 'local pressure')
        datanow[1] = datanow[1] + train_df.get_value(i, 'sea pressure')
        datanow[2] = datanow[2] + train_df.get_value(i, 'total preci')
        datanow[3] = datanow[3] + train_df.get_value(i, 'hr1 preci')
        datanow[4] = datanow[4] + train_df.get_value(i, 'min10 preci')
        datanow[5] = datanow[5] + train_df.get_value(i, 'avg temp')
        datanow[6] = datanow[6] + train_df.get_value(i, 'max temp')
        datanow[7] = datanow[7] + train_df.get_value(i, 'min temp')
        datanow[8] = datanow[8] + train_df.get_value(i, 'avg humid')
        datanow[9] = datanow[9] + train_df.get_value(i, 'min humid')
        datanow[10] = datanow[10] + train_df.get_value(i, 'sun hours')

        count = count + 1
        if (count == 120):
            break
    #a = np.ones(len(datanow)) * 120
    #data_now1 = np.divide(datanow, a)
    train_data.append(datanow)
    datanow = np.zeros(11)
    count = 0

train_target = train_bloom_days[:-1]

print(np.shape(train_data))

datanow = np.zeros(11)
test_data = []
p = range(0, 5)
count = 0

for m in p:
    for i in test_df[test_df.year == test_years[m]].serial:
        datanow[0] = datanow[0] + test_df.get_value(i, 'local pressure')
        datanow[1] = datanow[1] + test_df.get_value(i, 'sea pressure')
        datanow[2] = datanow[2] + test_df.get_value(i, 'total preci')
        datanow[3] = datanow[3] + test_df.get_value(i, 'hr1 preci')
        datanow[4] = datanow[4] + test_df.get_value(i, 'min10 preci')
        datanow[5] = datanow[5] + test_df.get_value(i, 'avg temp')
        datanow[6] = datanow[6] + test_df.get_value(i, 'max temp')
        datanow[7] = datanow[7] + test_df.get_value(i, 'min temp')
        datanow[8] = datanow[8] + test_df.get_value(i, 'avg humid')
        datanow[9] = datanow[9] + test_df.get_value(i, 'min humid')
        datanow[10] = datanow[10] + test_df.get_value(i, 'sun hours')

        count = count + 1
        if (count == 120):
            break

    #a = np.ones(len(datanow)) * 120
    #data_now1 = np.divide(datanow, a)
    test_data.append(datanow)
    datanow = np.zeros(11)
    count = 0

test_target = days_true

#print(test_data)


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

scaler = StandardScaler()

#X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.20, random_state=0)

#X_train_scaled = scaler.fit(X_train).transform(X_train)
#X_test_scaled = scaler.fit(X_test).transform(X_test)
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.fit_transform(test_data)

mlp = MLPRegressor(max_iter = 500, solver='lbfgs', hidden_layer_sizes = (5, 5), alpha = 0.1, random_state = 8)
mlp.fit(train_data_scaled, train_target)

print(mlp.score(train_data_scaled, train_target))
print(mlp.score(test_data_scaled, test_target))

pred_day = mlp.predict(test_data_scaled)
print(test_target)
print(pred_day.astype(int))