import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Input
from keras.optimizers import SGD
from keras.utils import plot_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

x=np.load("x.npy")
y=np.load("y.npy")
deleted_index = np.where(y == "neutral")
deleted_index = np.concatenate([deleted_index,np.where(y == "fearful")],axis=1)
deleted_index = np.concatenate([deleted_index,np.where(y == "disgust")],axis=1)
x=np.delete(x, deleted_index, 0)
y=np.delete(y, deleted_index, 0) # Tahmin edilmeyecek duygular burada çıkarılıyor.

x=x[:,:32] # Veriseti incelendiğinde 180 feature'dan ilk 32'si kullanılarak da iyi sonuç alındığı gözlendi

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x) # Veriseti standardize edildi.


# pca_32 = PCA(n_components=32, random_state=42)
# X_scaled = pca_32.fit_transform(x_scaled)
# PCA ile boyut azaltımı


emotions = np.array(["neutral","calm","happy","sad","angry","fearful","disgust","surprised"])
emotions=np.delete(emotions,[0,5,6])
enc=np.zeros((len(y),len(emotions)))
for i in range(len(y)):
    ind=np.where(emotions==y[i])
    enc[i,ind]=1
y=enc
# Kategorik labellar'ın sayısal hale getirilmesi için One-Hot Encoding yapıldı.

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.25, random_state=42)


model = Sequential()
model.add(Dense(256,input_dim=x_train.shape[1],activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(y_train.shape[1],activation='sigmoid')) # İki gizli katmanlı sinir ağı oluşturuldu.


model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)
model.fit(x_train,y_train,batch_size=int(len(x_train)*0.9/1),epochs=500,verbose=0)

loss, accuracy = model.evaluate(x_test,y_test,verbose=0)

y_pred = model.predict_classes(x_test) # model eğitildi ve test verileri için tahminler oluşturuldu.

print(accuracy)


pred = []
for j in range(len(y_test)):

    ind2 = y_pred[j]
    i2 = emotions[int(ind2)]
    pred.append(i2)
pred = np.array(pred)
desired = []
for j in range(len(y_test)):
    index1 = np.argmax(y_test[j])
    desired.append(emotions[index1])
desired = np.array(desired)
# tahmin ve ger.ek labellar için Inverse One-Hot Encoding.

confusionMatrix=metrics.confusion_matrix(desired,pred,labels= emotions)
df = pd.DataFrame(confusionMatrix, emotions, emotions)
fig4 = plt.figure()
fig4.suptitle("Confusion Matrix")
sn.set(font_scale=1.2)
sn.heatmap(df, annot=True, annot_kws={"size": 16}) # hata matrisi görselleştirildi.


con_mat=df.to_numpy()
correct=np.trace(con_mat)

print("\n MLP(2 hidden layer, Keras): {} of {} data were classified correctly.".format(correct, len(x_test)))
print(f"Accuracy: %{correct/len(y_test)*100}")

plt.show()