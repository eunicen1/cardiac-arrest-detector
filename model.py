import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

np.random.seed(91)

#setup
df = pd.read_csv('data2.csv', delimiter=',')
df = df.sample(frac=1)
arr = df.to_numpy()
splitarr = np.array_split(arr, 5, axis=0)
train = np.concatenate((splitarr[0], splitarr[4]), axis=0)
valid = splitarr[1]
test = splitarr[2]
a, b = train.shape
c, d = valid.shape
e, f = test.shape
Xtrain = train[:,:-1]
ytrain = train[:,b-1]
Xvalid = valid[:,:-1]
yvalid = valid[:,d-1]
Xtest = test[:,:-1]
ytest = test[:,f-1]

#cross-validation for KNN
kscores = []
for k in range(1,c):
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    knnModel = KNeighborsClassifier(k)
    scores = cross_val_score(knnModel, Xvalid, yvalid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    kscores.append(np.mean(np.abs(scores)))

kscores = np.concatenate((np.arange(1,c).reshape(c-1,1), np.array(kscores).reshape(c-1,1)),axis=1)
print(kscores)
kscores = np.array([kscores[i,:] for i in range(c-1) if np.isnan(kscores[i,1]) == False])
print(kscores)
kscore = kscores[np.argmin(kscores[:,1]),0] #select best kscore
print('best score is @ seed === 91: ', kscore)
#training
nnModel = KNeighborsClassifier(n_neighbors=int(kscore))
nnModel.fit(Xtrain, ytrain)
# plot_classifier(nnModel, Xtrain,ytrain)
ypred = nnModel.predict(Xtest)
acc = accuracy_score(ytest, ypred)
cm = confusion_matrix(ytest, ypred)
print(acc, cm)
