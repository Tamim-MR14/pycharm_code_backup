# Problem 2-1
import pandas as pd
import numpy as np
import math as mt
import matplotlib.pyplot as plt
from sklearn import metrics

df = pd.read_csv("modified tokyo.csv")
D = []
year = [i for i in range(1961, 2018)]

for i in range(1961, 2018):
    df2 = df.loc[df['year'] == i]
    df2.index = range(len(df2))
    tf = 0
    for j in range(90):
        tf = tf + df2.iloc[j, 9]
    tf = tf / 90
    fi = 35.67
    L = 4
    Dj = 136.75 - (7.689 * fi) + (0.133 * fi ** 2) - 1.307 * mt.log(L) + 0.144 * tf + 0.285 * tf ** 2
    D.append(Dj)

D = [int(round(i)) for i in D]
print("The calculated Dj s are:", D)
plt.plot(year, D)
plt.ylabel("Last day of Hibernation phase")
plt.xlabel("Year")
plt.show()

Ts = (17 + 273)
DTS = []
for i in range(1961, 2018):
    DTS.append([])
    if i in [1966, 1971, 1985, 1994, 2008]:
        DTS[i - 1961].append([])
        continue

    df2 = df.loc[df['year'] == i]
    df2.index = range(len(df2))
    bdi = df2.loc[df2['bloom'].isin([1])].index.tolist()
    for j in range(5, 41):
        E = j * 4184
        dts = 0
        for k in range(D[i - 1961], bdi[0] + 1):
            T = (df2.iloc[k, 9]) + 273
            dts = dts + np.exp((E * (T - Ts)) / (8.314 * T * Ts))
        DTS[i - 1961].append(dts)

DTSmean = []
for i in range(36):
    mean = 0
    for j in range(1961, 2018):
        if j in [1966, 1971, 1985, 1994, 2008]:
            continue
        mean = mean + DTS[j - 1961][i]
    mean = mean / 52
    DTSmean.append(mean)

print(DTSmean)
E = [E for E in range(5, 41)]
for i in range(1961, 2018):
    if i in [1966, 1971, 1985, 1994, 2008]:
        continue
    b = DTS[i - 1961]
    plt.plot(E, b)
plt.plot(E, DTSmean)
plt.show()

meansquareerror = []
for i in range(5, 41):
    squareerror = 0
    Ea = i * 4184

    for j in [1966, 1971, 1985, 1994, 2008]:
        df2 = df.loc[df['year'] == j]
        df2.index = range(len(df2))
        tf = 0
        for k in range(90):
            tf = tf + df2.iloc[k, 9]
        tf = tf / 90
        fi = 35.67
        L = 4
        Dj = 136.75 - (7.689 * fi) + (0.133 * fi ** 2) - 1.307 * mt.log(L) + 0.144 * tf + 0.285 * tf ** 2
        Dj = int(round(Dj))
        d = Dj - 1
        dts = 0
        while dts <= (DTSmean[i - 5]):
            T = (df2.iloc[d, 9]) + 273
            dts = dts + np.exp((Ea * (T - Ts)) / (8.314 * T * Ts))
            d = d + 1
        pbd = d
        bdi = df2.loc[df2['bloom'].isin([1])].index.tolist()
        abd = bdi[0]
        squareerror = squareerror + (abd - pbd) ** 2

    squareerror = squareerror / 5
    meansquareerror.append(squareerror)

plt.plot(E, meansquareerror)
plt.show()

print("The calculated DTSmean for Ea=28 is:", DTSmean[28-5])
abd=[]
pbd=[]
error=[]
Ea=28*4184
for i in [1966,1971,1985,1994,2008]:
    Dj=D[i-1961]
    d=Dj-1
    df2=df.loc[df['year'] == i]
    df2.index=range(len(df2))
    dts=0
    while dts<=(DTSmean[23]) :
        T=(df2.iloc[d,9])+273
        dts=dts+np.exp((Ea*(T-Ts))/(8.314*T*Ts))
        d=d+1
    pbd.append(d)
    bdi=df2.loc[df2['bloom'].isin([1])].index.tolist()
    abd.append(bdi[0])
    error.append(bdi[0]-d)
print("The errors are:", error)
rscore=metrics.r2_score(abd,pbd)
print("The R^2 score using Ea=28 kcal and corresponding Dj and DTSmean",rscore)