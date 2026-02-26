# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. Import the necessary libraries
2. Load the dataset using sklearn.datasets()
3. Convert the dataset into a dataframe
4. Define the input and target variable
5. Split the dataset into training and testing data
6. Train the model using SGDClassifier(),.fit() and predict using .predict()
7. Measure the accuracy of the model using accuracy_score() and confusion_matrix() 

## Program:
```
      import pandas as pd
      from sklearn.datasets import load_iris
      from sklearn.linear_model import SGDClassifier
      from sklearn.model_selection import train_test_split
      from sklearn.metrics import accuracy_score,confusion_matrix
      import matplotlib.pyplot as plt
      import seaborn as sns
      
      iris=load_iris()
      
      df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
      df['target']=iris.target
      
      print(df.head())
      print(df.tail())
      
      x=df.drop('target',axis=1)
      y=df['target']
      
      x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
      
      sgd=SGDClassifier(max_iter=1000,tol=1e-3)
      sgd.fit(x_train,y_train)
      y_pred=sgd.predict(x_test)
      
      acc=accuracy_score(y_test,y_pred)
      print(acc)
      cm=confusion_matrix(y_test,y_pred)
      print(cm)

Program to implement the prediction of iris species using SGD Classifier.
Developed by:Sangeeth M 
RegisterNumber:25004402

```

## Output:
![image](https://github.com/user-attachments/assets/8e8ff478-c573-41e1-a66d-9f1f363d9e40)
![image](https://github.com/user-attachments/assets/73994dea-1c67-4a5c-a45c-49c22a69a3a5)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
