from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plot
from sklearn import linear_model,metrics
data=pd.read_csv('bostonhousing_datasets.csv',header =None)
clmns=data.iloc[0:1]

df=pd.DataFrame(data[1:])
df.columns=data.iloc[0]
print(df)

X=df.iloc[:,:3]
Y=df.iloc[:,3:]

# Train Test Split
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, 
                                                    random_state=1)
# Formation of Learning Model
mod=linear_model.LinearRegression()

#Training of Learning model
mod.fit(X_train,Y_train)

# Printing of Trained Co-efficients
print('\nCoefficients : ',mod.coef_)

# Printing The Score of Accuracy
print('\nVariance Score : ',format(mod.score(X_test,Y_test)))

# Data Plotting

plot.subplot(1,2,2)
plot.title('Training-Testing dataset')
plot.xlabel('Room per man')
plot.ylabel('Median Value of room')
y1=plot.scatter(X_train['RM'],Y_train['MEDV'] , alpha=1.0,marker="*",color='c')
y2=plot.scatter(X_test['RM'],Y_test['MEDV'] , alpha=1.0,marker=",",color='r')

plot.subplot(1,2,1)
y3=plot.scatter(df['RM'], df['MEDV'], alpha=1.0,marker="*")

plot.title('Original dataset')
plot.xlabel('Room per man')
plot.ylabel('Median Value of roomprices')


plot.show()
