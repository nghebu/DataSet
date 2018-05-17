import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
#import DATASET 1
TrainData1=pandas.read_csv('TrainData1.txt',sep="\t",header=None,engine='python')
TrainLabel1=pandas.read_csv('TrainLabel1.txt',sep="\t",header=None,engine='python')
TestData1=pandas.read_csv('TestData1.txt',sep="\t",header=None,engine='python')
#Fill Missing number, Mean Value method
TrainData1=TrainData1.replace(1.0000000000000001e+99,numpy.NaN)
TrainData1.fillna(TrainData1.mean(), inplace=True)
#Export
TrainData1.to_csv('D:\DataSet\TrainData1Fixed.txt',index=False, sep="\t", header=None)
FixedTrain1=pandas.read_csv('TrainData1Fixed.txt',sep="\t",header=None,engine='python',index_col = False)
# Splitting
train, test, train_labels, test_labels = train_test_split(FixedTrain1,TrainLabel1, test_size=0.353, random_state=40)
# Train our classifier
gnb = GaussianNB()
model = gnb.fit(train,train_labels.values.ravel())
#Statistic information
#print(TrainData1.describe())
#print(FixedTrain.describe())
# Make predictions
prediction = gnb.predict(test)
numpy.savetxt('D:\DataSet\Test_Label1_Predicted.txt',prediction, delimiter='\t')
print(accuracy_score(test_labels, prediction))

#import DATASET 2
TrainData1=pandas.read_csv('TrainData2.txt',sep="\t",header=None,engine='python')
TrainLabel1=pandas.read_csv('TrainLabel2.txt',sep="\t",header=None,engine='python')
TestData1=pandas.read_csv('TestData2.txt',sep="\t",header=None,engine='python')
#Fill Missing number, Mean Value method
TrainData1=TrainData1.replace(1.0000000000000001e+99,numpy.NaN)
TrainData1.fillna(TrainData1.mean(), inplace=True)
#Export
TrainData1.to_csv('D:\DataSet\TrainData2Fixed.txt',index=False, sep="\t", header=None)
FixedTrain1=pandas.read_csv('TrainData2Fixed.txt',sep="\t",header=None,engine='python',index_col = False)
# Splitting
train, test, train_labels, test_labels = train_test_split(FixedTrain1,TrainLabel1, test_size=0.353, random_state=40)
# Train our classifier
gnb = GaussianNB()
model = gnb.fit(train,train_labels.values.ravel())
#Statistic information
#print(TrainData1.describe())
#print(FixedTrain.describe())
# Make predictions
prediction = gnb.predict(test)
numpy.savetxt('D:\DataSet\Test_Label2_Predicted.txt',prediction, delimiter='\t')
print(accuracy_score(test_labels, prediction))

#import DATASET 3
TrainData1=pandas.read_csv('TrainData3.txt',sep="\t",header=None,engine='python')
TrainLabel1=pandas.read_csv('TrainLabel3.txt',sep="\t",header=None,engine='python')
TestData1=pandas.read_csv('TestData3.txt',sep="\t",header=None,engine='python')
#Fill Missing number, Mean Value method
TrainData1=TrainData1.replace(1.0000000000000001e+99,numpy.NaN)
TrainData1.fillna(TrainData1.mean(), inplace=True)
#Export
TrainData1.to_csv('D:\DataSet\TrainData3Fixed.txt',index=False, sep="\t", header=None)
FixedTrain1=pandas.read_csv('TrainData3Fixed.txt',sep="\t",header=None,engine='python',index_col = False)
# Splitting
train, test, train_labels, test_labels = train_test_split(FixedTrain1,TrainLabel1, test_size=0.353, random_state=60)
# Train our classifier
gnb = GaussianNB()
model = gnb.fit(train,train_labels.values.ravel())
#Statistic information
#print(TrainData1.describe())
#print(FixedTrain.describe())
# Make predictions
prediction = gnb.predict(test)
numpy.savetxt('D:\DataSet\Test_Label3_Predicted.txt',prediction, delimiter='\t')
print(accuracy_score(test_labels, prediction))

#import DATASET 4
TrainData1=pandas.read_csv('TrainData4.txt',sep="\t",header=None,engine='python')
TrainLabel1=pandas.read_csv('TrainLabel4.txt',sep="\t",header=None,engine='python')
TestData1=pandas.read_csv('TestData4.txt',sep="\t",header=None,engine='python')
#Fill Missing number, Mean Value method
TrainData1=TrainData1.replace(1.0000000000000001e+99,numpy.NaN)
TrainData1.fillna(TrainData1.mean(), inplace=True)
#Export
TrainData1.to_csv('D:\DataSet\TrainData4Fixed.txt',index=False, sep="\t", header=None)
FixedTrain1=pandas.read_csv('TrainData4Fixed.txt',sep="\t",header=None,engine='python',index_col = False)
# Splitting
train, test, train_labels, test_labels = train_test_split(FixedTrain1,TrainLabel1, test_size=0.1, random_state=10)
# Apply Label_Encoding
le = preprocessing.LabelEncoder()
train=train.apply(le.fit_transform)
test=test.apply(le.fit_transform)
# Train our classifier
gnb = GaussianNB()
model = gnb.fit(train,train_labels.values.ravel())
# Make predictions
prediction = gnb.predict(test)
numpy.savetxt('D:\DataSet\Test_Label4_Predicted.txt',prediction, delimiter='\t')
print(accuracy_score(test_labels, prediction))

#import DATASET 5
TrainData1=pandas.read_csv('TrainData5.txt',sep="\t",header=None,engine='python')
TrainLabel1=pandas.read_csv('TrainLabel5.txt',sep="\t",header=None,engine='python')
TestData1=pandas.read_csv('TestData5.txt',sep="\t",header=None,engine='python')
#Fill Missing number, Mean Value method
TrainData1=TrainData1.replace(1.0000000000000001e+99,numpy.NaN)
TrainData1.fillna(TrainData1.mean(), inplace=True)
#Export
TrainData1.to_csv('D:\DataSet\TrainData5Fixed.txt',index=False, sep="\t", header=None)
FixedTrain1=pandas.read_csv('TrainData5Fixed.txt',sep="\t",header=None,engine='python',index_col = False)
# Splitting
train, test, train_labels, test_labels = train_test_split(FixedTrain1,TrainLabel1, test_size=0.1, random_state=51)
# Apply Label_Encoding
le = preprocessing.LabelEncoder()
train=train.apply(le.fit_transform)
test=test.apply(le.fit_transform)
# Train our classifier
gnb = GaussianNB()
model = gnb.fit(train,train_labels.values.ravel())
# Make predictions
prediction = gnb.predict(test)
numpy.savetxt('D:\DataSet\Test_Label5_Predicted.txt',prediction, delimiter='\t')
print(accuracy_score(test_labels, prediction))
