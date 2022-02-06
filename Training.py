import numpy as np
import cv2
import pickle
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.naive_bayes import ComplementNB
from sklearn  import metrics 


with open("baseIndex.ss","rb") as f:
    x=pickle.load(f)
    y=pickle.load(f)
    
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
model=KNeighborsClassifier(15)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

model.fit(x,y)
# save model 

with open("modelKnn","wb") as f:
    pickle.dump(model,f)
