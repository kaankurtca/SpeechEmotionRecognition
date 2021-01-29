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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB,CategoricalNB



x=np.load("x.npy")
y=np.load("y.npy")
deleted_index = np.where(y == "neutral")
deleted_index = np.concatenate([deleted_index,np.where(y == "fearful")],axis=1)
deleted_index = np.concatenate([deleted_index,np.where(y == "disgust")],axis=1)
x=np.delete(x, deleted_index, 0)
y=np.delete(y, deleted_index, 0) # Tahmin edilmeyecek duygular burada çıkarılıyor.

x=x[:,:36]
# Veriseti incelendiğinde 180 feature'dan ilk 32'si kullanılarak da iyi sonuç alındığı gözlendi
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x) # Veriseti standardize edildi.


pca_32 = PCA(n_components=32, random_state=42)
X_scaled = pca_32.fit_transform(x_scaled)
# PCA ile boyut azaltımı

emotions = np.array(["calm","happy","sad","angry","surprised"])
enc=np.zeros((len(y), len(emotions)))
for i in range(len(y)):
    ind=np.where(emotions == y[i])
    enc[i,ind]=1
y_enc=enc
# One-hot Encoding (Decision Tree ve Random Forest için)

enc2=np.zeros(len(y))
for i in range(len(y)):
    ind = np.where(emotions==y[i])
    enc2[i] = ind[0]
y_NaiveBayes=enc2
# Label Encoding (Naive Bayes için)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_NaiveBayes, test_size=0.3, random_state=42)
classifier = GaussianNB()
classifier.fit(x_train, y_train)
y_pred_NB = classifier.predict(x_test)

y_test1 = y_test.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_enc, test_size=0.3, random_state=42)

regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(x_train, y_train)
y_pred_rf = regressor.predict(x_test)

classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)
y_pred_dt = classifier.predict(x_test)

pred_rf , pred_dt, desired = [], [], []
for j in range(len(y_test)):
    index1 = np.argmax(y_test[j])
    desired.append(emotions[index1])
    index2 = np.argmax(y_pred_rf[j])
    pred_rf.append(emotions[index2])
    index3 = np.argmax(y_pred_dt[j])
    pred_dt.append(emotions[index3])
pred_rf, pred_dt, desired = np.array(pred_rf), np.array(pred_dt), np.array(desired)

pred_NB = []
for j in range(len(y_test)):
    ind_NB = y_pred_NB[j]
    i2_NB = emotions[int(ind_NB)]
    pred_NB.append(i2_NB)
pred_NB= np.array(pred_NB)

#tahmin edilen labellar'a Inverse One-hot ve Label encoding uygulandı.

# aşağıda, 3 sınıflandırıcı için de test hata matrisleri görselleştirildi.

confusionMatrix1=metrics.confusion_matrix(desired,pred_rf,labels= emotions)
df1 = pd.DataFrame(confusionMatrix1, emotions, emotions)
fig4 = plt.figure()
fig4.suptitle("Confusion Matrix (Random Forest)")
sn.set(font_scale=1)
sn.heatmap(df1, annot=True, annot_kws={"size": 16})
con_mat1=df1.to_numpy()
correct=np.trace(con_mat1)
print("\n Random Forest: {} of {} data were classified correctly.".format(correct, len(x_test)))
print(f"Accuracy: %{correct/len(y_test)*100}")

confusionMatrix2=metrics.confusion_matrix(desired,pred_dt,labels= emotions)
df2 = pd.DataFrame(confusionMatrix2, emotions, emotions)
fig5 = plt.figure()
fig5.suptitle("Confusion Matrix (Decision Tree)")
sn.set(font_scale=1)
sn.heatmap(df2, annot=True, annot_kws={"size": 16})
con_mat2=df2.to_numpy()
correct=np.trace(con_mat2)
print("\n Decision Tree: {} of {} data were classified correctly.".format(correct, len(x_test)))
print(f"Accuracy: %{correct/len(y_test)*100}")

confusionMatrix3=metrics.confusion_matrix(desired,pred_NB,labels= emotions)
df3 = pd.DataFrame(confusionMatrix3, emotions, emotions)
fig6 = plt.figure()
fig6.suptitle("Confusion Matrix (Naive Bayes)")
sn.set(font_scale=1)
sn.heatmap(df3, annot=True, annot_kws={"size": 16})
con_mat3=df3.to_numpy()
correct=np.trace(con_mat3)
print("\n Naive Bayes: {} of {} data were classified correctly.".format(correct, len(x_test)))
print(f"Accuracy: %{correct/len(y_test)*100}")



plt.show()


