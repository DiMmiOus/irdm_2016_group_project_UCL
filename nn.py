import numpy as np
import pandas as pd


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD


w1 = pd.read_table("weights.out",sep=',', header=None)
w2 = pd.read_table("weights2.out",sep=',', header=None)
w3 = pd.read_table("weights3.out",sep=',', header=None)
w4 = pd.read_table("weights4.out",sep=',', header=None)
U_train =pd.read_table("utrain.out",sep=',', header=None)
print(type(U_train))


U_train_1 = U_train.values

new_U_train = np.ones((750,1683))
new_U_train[:,:-1] = U_train_1
def f(x):
    return float(x)/5
f = np.vectorize(f)
result_array = f(new_U_train)
print(np.isnan(np.min(result_array)))
print(result_array)

model = Sequential()
# 1st layer
model.add(Dense(51,input_dim = 1683,weights=[w1,np.ones(51)],activation='sigmoid'))
model.add(Dropout(0.1))# dropout
# 2nd layer
model.add(Dense(26,input_dim =51,weights=[w2,np.ones(26)],activation='sigmoid'))
model.add(Dropout(0.1))
# 3rd layer
model.add(Dense(51,input_dim = 26,weights=[w3,np.ones(51)],activation='sigmoid'))
model.add(Dropout(0.1))
# 4th layer
model.add(Dense(1683,input_dim = 51,weights=[w4,np.ones(1683)],activation='sigmoid'))
model.add(Dropout(0.1))
#
#
sgd = SGD(lr=0.03, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer=sgd)


model.fit(result_array,result_array, nb_epoch=20, batch_size=51,validation_split=0.2)
