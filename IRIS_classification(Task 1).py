import matplotlib.pyplot as plt
#loading irisdataset from sklearn datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
#storing dataset in a variable to access it easily
iris=load_iris()

#applying transpose function on "data" key values 
features=iris.data.T

#storing each attribute's values in specific variable
sepal_length=features[0]
sepal_width=features[1]
petal_length=features[2]
petal_width=features[3]

features_names=iris.feature_names
#storing each label of feature in specific variable
sepal_length_label=features_names[0]
sepal_width_label=features_names[1]
petal_length_label=features_names[2]
petal_width_label=features_names[3]

target_names=iris.target_names

s=plt.scatter(sepal_length,sepal_width , c=iris.target)

plt.xlabel(sepal_length_label)
plt.ylabel(sepal_width_label)
plt.legend(s.legend_elements()[0],target_names,title="Class")
plt.title("Iris Classification")

plt.show()

X_train,x_test,Y_train,y_test=train_test_split(iris['data'],iris['target'],random_state=0)#here capital x and y means that the data is for training and small x and y means that data is for testing and iris data will be given as input and iris target is given as output i.e. model will predict the value of target on the basis of iris data
knn=KNeighborsClassifier(n_neighbors=1)# k value is given that is, target value will be decided on the basis of nearest neighbour here only 1 nearest neighbour will be considered
knn.fit(X_train,Y_train)# now we are shaping the k model according to our needs that is we are training our k model on our data by feeding the data
#now to test the accuracy of our model we will be using score funcion and what it will do is that it will take x_test and y_test data and then use the knn model which we have trained to predict the utput or target value from test data set(x_test)and compare its results with the y_test value  and then tell the accuracy that how close the predicted result is 
acc=knn.score(x_test,y_test)
print("Accuracy level:",acc)
feature_values=[]
feature_values1=[]
for i in features_names:
    x_new=float(input(f"{i}:"))
    feature_values.append(x_new)
    feature_values1.append(feature_values)

predict=knn.predict(feature_values1)
predict0=predict[0]
if predict0==[0]:
    print("Class:Setosa",predict0)
elif predict0==[1]:
    print("Class:versicolor",predict0)
else:
    print("Class:Virginica",predict0)



