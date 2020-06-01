from data_preprocessing import *
from tf_idf import *
from word2vec_model import *
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
# path_train=r"D:\20192\machine_learning\data\train.crash"
# path_test=r"D:\20192\machine_learning\data\test.crash"
# #get data and processing
# data_train,label_train=get_data_train(path_train)

# data_test = get_data_test(path_test)

# all_data=data_train+data_test
# s=set()
# for line in all_data:
#   for w in line:
#     s.add(w)

# #train model word2vec
# word2vec_model=start(all_data) 
# #convert word to vector
# X=tf_idf(data_train,word2vec_model)
# y=[]
# for i in label_train:
#   y.append(int(i))
# X=np.array(X)
# y=np.array(y)
# np.save("best_data",X)
# np.save("best_label",y)
X = np.load("best_data.npy")
y = np.load("best_label.npy")
#split train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#SVM model

svm = SVC()
param = {'C': [6,7,8] , 'kernel': ['rbf']}

gs = GridSearchCV(estimator=svm, param_grid=param, cv=3, n_jobs=4)
gs.fit(X_train, y_train)
print(gs.score(X_train, y_train))

print(gs.score(X_test, y_test))
filename = 'last_model.sav'
joblib.dump(gs, filename)
