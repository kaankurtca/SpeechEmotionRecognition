import pickle
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
import numpy as np
from sklearn.neighbors import KNeighborsClassifier




x=np.load("x.npy")
y=np.load("y.npy")
deleted_index = np.where(y == "neutral")
deleted_index = np.concatenate([deleted_index,np.where(y == "fearful")],axis=1)
deleted_index = np.concatenate([deleted_index,np.where(y == "disgust")],axis=1)
x=np.delete(x, deleted_index, 0)
y=np.delete(y, deleted_index, 0) # Tahmin edilmeyecek duygular burada çıkarılıyor.

x=x[:,:32] # Veriseti incelendiğinde 180 feature'dan ilk 32'si kullanılarak da iyi sonuç alındığı gözlendi

# pca_32 = PCA(n_components=32, random_state=42)
# X_scaled = pca_32.fit_transform(x)
# PCA ile boyut azaltımı

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x) # Veriseti standardize edildi.

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.25, random_state=42)

model=SVC(kernel='rbf',class_weight='balanced')
model.fit(x_train,y_train)
y_pred=model.predict(x_test) # model eğitildi ve test verileri için tahminler oluşturuldu.

accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print(f"Accuracy: %{accuracy*100}")

emotions = ["calm","happy","sad","angry","surprised"]

confusionMatrix=metrics.confusion_matrix(y_test, y_pred,labels= emotions)
df = pd.DataFrame(confusionMatrix, emotions, emotions)
fig4 = plt.figure()
fig4.suptitle("Confusion Matrix")
sn.set(font_scale=1.4)
sn.heatmap(df, annot=True, annot_kws={"size": 16}) # Hata matrisi görselleştirildi.

con_mat=df.to_numpy()
correct=np.trace(con_mat)

print("\nSupport Vector Machines: {} of {} data were classified correctly.".format(correct, len(x_test)))
print(f"Accuracy: %{correct/len(y_test)*100}")


filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

plt.show()