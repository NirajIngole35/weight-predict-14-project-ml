
import pandas as pd
import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from sklearn .model_selection import train_test_split
from sklearn .linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn .metrics import classification_report,confusion_matrix,accuracy_score,r2_score


# Step 1 - Load data and understand
ds = pd.read_csv(r"C:\Users\HP\Desktop\project of data science\weight predict\weight-height.csv")
print(ds)
print (ds.head(10))
print (ds.tail(10))
print (ds.describe())
print (ds.isnull().sum())
print (ds.shape)
print('##################################################')
#step 2-convert categoricaldata tinto numerical data

ds['Gender'].replace('Female',0, inplace=True)
ds['Gender'].replace('Male',1, inplace=True)
x = ds.iloc[:, :-1].values
y = ds.iloc[:, 2].values
print(x)
print('##################################################')

# Step 3 - visualize



plt.scatter(x=ds["Gender"],y=ds["Height"],color="green")
plt.grid(True)

plt.show
print('##################################################')
#sstep 4-splite

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=(0.2),random_state=0)

print('##################################################')
#step 5 train
model=LinearRegression()

model.fit(x_train,y_train)



#step 6 preadic

pre=model.predict(x_test)
print(pre)


#step 7 output


print("r2_score of model is :{0}%".format(r2_score(y_test,pre)*100))
