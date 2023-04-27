# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## Aim:
To write a program to implement the  Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values.

## Program:
```
/*
Program to implement the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Smriti M
RegisterNumber: 212221040157
*/
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x

y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

## Placement data 

![s1](https://user-images.githubusercontent.com/113674204/231648649-33f7653d-1911-4ba3-91da-3e9356a138f2.png)

## Salary data

![s2](https://user-images.githubusercontent.com/113674204/231648766-c04dabd7-2e6a-4468-8a81-f06cfa522da4.png)

## Checking null function

![s3](https://user-images.githubusercontent.com/113674204/231648812-72501394-b313-4fdd-ac86-a24d01d8c91d.png)

## Data duplicate

![s4](https://user-images.githubusercontent.com/113674204/231648864-7b407d77-f8f7-49b2-a05d-281b5b945041.png)

## Print data

![s5](https://user-images.githubusercontent.com/113674204/231648946-47753b98-57e8-451e-abbd-276008a522d8.png)

## Data status

![s7](https://user-images.githubusercontent.com/113674204/231649135-14c5c342-5e42-455a-ac6f-27766be40fa2.png)

## Y-prediction array

![s8](https://user-images.githubusercontent.com/113674204/231649242-a6327390-6a81-4dbf-b9b3-31b70d4c727d.png)

## Accuracy value

![s9](https://user-images.githubusercontent.com/113674204/231649297-ce66d699-a122-47de-9c44-d8d4f94010e1.png)

## Confusion array

![s10](https://user-images.githubusercontent.com/113674204/231649352-cbff8b74-3fbc-46bf-95b9-7bd8139e9de3.png)

## Classification report

![s11](https://user-images.githubusercontent.com/113674204/231649409-913bd15b-c552-4690-a584-570be55c729d.png)

## Prediction of LR

![s12](https://user-images.githubusercontent.com/113674204/231649475-158f4706-b083-4431-879f-5642bd6903af.png)

## Result:
Thus the program to implement the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
