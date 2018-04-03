from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
names=['Bob','Jessica','Merry','John','Mel']
births=[968,155,77,578,973]
BabyDataSet=list(zip(names,births))
print(BabyDataSet)
df=pd.DataFrame(data=BabyDataSet, columns=['Names','Births'])
print(df)
df.to_csv('births1980.csv',index=False,header=False)
df=pd.read_csv('births1980.csv',header=None)
print(df)
df=pd.read_csv('births1980.csv',names=['Names','Births'])

import os
os.remove('births1980.csv')


print(df.Births.dtype)
Sorted=df.sort_values(['Births'], ascending=False)
print(Sorted.head(1))

print(df['Births'].max( ))
