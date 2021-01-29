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
from MLP_Class_fromScratch import MultiP
from sklearn.preprocessing import OneHotEncoder

x=np.load("x.npy")
y=np.load("y.npy")
deleted_index = np.where(y == "neutral")
deleted_index = np.concatenate([deleted_index,np.where(y == "fearful")],axis=1)
deleted_index = np.concatenate([deleted_index,np.where(y == "disgust")],axis=1)
x=np.delete(x, deleted_index, 0)
y=np.delete(y, deleted_index, 0) # Tahmin edilmeyecek duygular burada çıkarılıyor.

x=x[:,:32]
# Veriseti incelendiğinde 180 feature'dan ilk 32'si kullanılarak da iyi sonuç alındığı gözlendi

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# pca_32 = PCA(n_components=32, random_state=42)
# X_scaled = pca_32.fit_transform(x)
# PCA ile boyut azaltımı

emotions = np.array(["calm","happy","sad","angry","surprised"])
enc=np.zeros((len(y),len(emotions)))
for i in range(len(y)):
    ind=np.where(emotions==y[i])
    enc[i,ind]=1
y=enc
# One-hot Encoding

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.25, random_state=42)

mlp = MultiP(x_train.shape[1],130,130,y.shape[1])
mlp.train(x_train,y_train,500,0.1)  # Eğitim uzun sürmektedir.

y_pred=np.zeros((len(y_test), y_test.shape[1]))
for k in range(len(x_test)):
    output = np.around(mlp.feedForward(x_test[k]).reshape(-1, 1), 3)  # tahmin edilen çıkış
    y_pred[k, :] = output.reshape(1, -1)



pred, desired = [], []
for j in range(len(y_test)):
    index1 = np.argmax(y_test[j])
    desired.append(emotions[index1])
    index2 = np.argmax(y_pred[j])
    pred.append(emotions[index2])
pred, desired = np.array(pred), np.array(desired)
# Inverse One-Hot Encoding

confusionMatrix=metrics.confusion_matrix(desired,pred,labels= emotions)
df = pd.DataFrame(confusionMatrix, emotions, emotions)
fig4 = plt.figure()
fig4.suptitle("Confusion Matrix")
sn.set(font_scale=1.4)
sn.heatmap(df, annot=True, annot_kws={"size": 16})
# hata matrisi görselleştirildi.

con_mat=df.to_numpy()
correct=np.trace(con_mat)

print("\nMLP(2 hidden layer): {} of {} data were classified correctly.".format(correct, len(x_test)))
print(f"Accuracy: %{correct/len(y_test)*100}")


plt.show()

