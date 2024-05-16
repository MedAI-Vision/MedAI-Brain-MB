import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



data = pd.read_csv("./data.csv")

train = data.loc[(data["fold"] == 'train')]
test = data.loc[(data["fold"] == 'test')]

features = ['', #features need to be analyse
'molecular']
test = test[features]

plt.figure(figsize=(25,10))
plt.subplots_adjust(left=0.14,right=0.7,bottom=0.14,top=0.88)
for i in range(1):
    plt.subplot(1,1,i+1)
    g = sns.boxplot(x=test['molecular'],y=test.iloc[:,i])
    plt.xlabel(str(features[i]))
    plt.ylabel('')

plt.show()
