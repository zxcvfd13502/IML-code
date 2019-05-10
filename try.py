import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,LeakyReLU,BatchNormalization,Conv1D,MaxPooling1D,Flatten
from keras.optimizers import SGD
from keras.utils import to_categorical

input_dim=120
output_dim=5
droupout_rate=0.5
neurons=512
epochs=500
batch_size=2048
def get_model(input_dim=120,output_dim=5,droupout_rate=0.5,neurons=300,epochs=300,batch_size=10000):
    model=Sequential()
    model.add(Conv1D(256, 10, activation='relu', input_shape=(input_dim, 1)))
    model.add(Conv1D(256, 10, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Dense(neurons))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(Flatten())
    model.add(Dense(neurons))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim,activation='softmax'))
    return model
data_train = pd.read_hdf("train.h5", key="train")
test = pd.read_hdf("test.h5", key="test")
X_train=data_train.iloc[:,1:]
X_train=np.array(X_train)
Y_train=data_train.iloc[:,0]
Y_train=np.array(Y_train)
Y_train=to_categorical(Y_train)
ss = StandardScaler()
#ss=MinMaxScaler()
X_train = ss.fit_transform(X_train)
X_train=X_train[:,:,np.newaxis]
print(X_train.shape)

#selector=SelectKBest(mutual_info_classif, k=input_dim)
#X_train = selector.fit_transform(X_train, Y_train)
#ss = StandardScaler()
#ss=MinMaxScaler()
#X_train = ss.fit_transform(X_train)


model=get_model(input_dim=120,output_dim=5,neurons=512)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(X_train, Y_train,epochs=epochs,batch_size=batch_size,validation_split=0.1,shuffle=True,class_weight = 'auto')
