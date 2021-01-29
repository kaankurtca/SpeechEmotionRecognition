import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

x=np.load("x.npy")
y=np.load("y.npy")
deleted_index = np.where(y == "neutral")
deleted_index = np.concatenate([deleted_index,np.where(y == "fearful")],axis=1)
deleted_index = np.concatenate([deleted_index,np.where(y == "disgust")],axis=1)
x=np.delete(x, deleted_index, 0)
y=np.delete(y, deleted_index, 0) # Tahmin edilmeyecek duygular burada çıkarılıyor.



x_newtest=np.load("x_newtest.npy").reshape(1,-1)

# pca_32 = PCA(n_components=32, random_state=42)
# X_scaled = pca_32.fit_transform(x)
# x_newtest = pca_32.fit_transform(x_newtest)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_newtest = x_newtest[:,:32]
x_newtest = scaler.fit_transform(x_newtest)

print(x_newtest.shape)



emotions = np.array(["calm","happy","sad","angry","surprised"])
enc=np.zeros((len(y),len(emotions)))
for i in range(len(y)):
    ind=np.where(emotions==y[i])
    enc[i,ind]=1
y=enc

model = pickle.load(open('finalized_model.sav', 'rb'))

y_pred=model.predict(x_newtest)

print(y_pred)


