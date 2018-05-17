import numpy
import pandas
from fancyimpute import KNN    
#import DATASET 1
MissingData=pandas.read_csv('MissingData1.txt',sep="\t",header=None,engine='python')
#Fill Missing number, Mean Value method
MissingData=MissingData.replace(1.0000000000000001e+99,numpy.NaN)
MissingData = KNN(k=3).complete(MissingData)
prediction = pandas.DataFrame(MissingData).to_csv('MissingData1Filled.txt',index=False, sep="\t", header=None)

#import DATASET 2
MissingData=pandas.read_csv('MissingData2.txt',sep="\t",header=None,engine='python')
#Fill Missing number, Mean Value method
MissingData=MissingData.replace(1.0000000000000001e+99,numpy.NaN)
MissingData = KNN(k=3).complete(MissingData)
prediction = pandas.DataFrame(MissingData).to_csv('MissingData2Filled.txt',index=False, sep="\t", header=None)

#import DATASET 3
MissingData=pandas.read_csv('MissingData3.txt',sep="\t",header=None,engine='python')
#Fill Missing number, Mean Value method
MissingData=MissingData.replace(1.0000000000000001e+99,numpy.NaN)
MissingData = KNN(k=3).complete(MissingData)
prediction = pandas.DataFrame(MissingData).to_csv('MissingData3Filled.txt',index=False, sep="\t", header=None)

