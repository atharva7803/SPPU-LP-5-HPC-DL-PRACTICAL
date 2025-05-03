import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Boston.csv')
df.head(10)

df.drop(columns=['Unnamed: 15','Unnamed: 16'],inplace=True)

df.drop(columns=['CAT. MEDV'],inplace=True)

df.isnull().sum()

df.info()

df.describe()

df.corr()['MEDV'].sort_values()

X = df.loc[:,['LSTAT','PTRATIO','RM']]
Y = df.loc[:,"MEDV"]
X.shape,Y.shape

# Preparing training and testing data set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=10)

# Normalizing training and testing dataset
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(128,input_shape=(3,),activation='relu',name='input'))
model.add(Dense(64,activation='relu',name='layer_1'))
model.add(Dense(1,activation='linear',name='output'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

model.fit(x_train,y_train,epochs=100,validation_split=0.05)

output = model.evaluate(x_test,y_test)

print(f"Mean Squared Error: {output[0]}"
      ,f"Mean Absolute Error: {output[1]}",sep="\n")

y_pred = model.predict(x=x_test)

print(*zip(y_pred,y_test))

