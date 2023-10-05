import serial
import numpy as np
import math
import scipy.stats as ss
import pandas as pd
from scipy.io import loadmat
from sklearn.tree import DecisionTreeClassifier



# Load the dataset
X_path = "C:/Users/user/Desktop/Hackerton/machine_learning/JustArrayData.mat"
y_path = "C:/Users/user/Desktop/Hackerton/machine_learning/JustArrayLabels.mat"

matData = loadmat(X_path)
matLabel = loadmat(y_path)

rawMatData = matData['JustArrayData']
rawMatLabel = matLabel['JustArrayLabels']

rawMatLabel[rawMatLabel == 2] = 0
rawMatLabel[rawMatLabel == 3] = 2

X = np.array(pd.DataFrame(rawMatData))
y = np.array(pd.DataFrame(rawMatLabel)).ravel()

DT_feature = [1, 2, 3, 4, 8, 17, 23, 25, 26, 28, 29, 30, 31, 35, 36]

DT_X = X[:, DT_feature]

# Train the model
DT = DecisionTreeClassifier(criterion="entropy", max_depth=23)
DT.fit(DT_X, y)

def aco(AccRawMedX):
    if (AccRawMedX>=-1) & (AccRawMedX<=1):
        return math.acos(AccRawMedX)
    else:
        return AccRawMedX
def iqr(data):
    q25,q75=np.quantile(data,0.25),np.quantile(data,0.75)
    result= q75-q25
    return result
def fft(record,number):
    func=[]
    func.append(record[0][int(number)])
    func.append(record[1][int(number)])
    func.append(record[2][int(number)])
    result=sum(np.fft.fft(func) / len(func))
    return result

ard=serial.Serial(port='COM5',baudrate=9600)

record=[]
for i in range(2):
    data=ard.readline()
    data=eval(data[:-2].decode())
    data1=list(map(float, data))
    data2=[]
    for i in data1:
      data2.append(i*0.001)
    AccRawMed = np.array([data2[0], data2[1], data2[2]])

    AccRawMed = ss.zscore(AccRawMed)

    record.append(AccRawMed)

while True:
    data=ard.readline()
    data=eval(data[:-2].decode())
    data1=list(map(float, data))
    data2=[]
    for i in data1:
      data2.append(i*0.001)
    data=data2

    AccRawMed = np.array([data[0], data[1], data[2]])
    #scaler = StandardScaler()
    AccRawMed=ss.zscore(AccRawMed)#scaler.fit_transform(AccRawMed)
    record.append(AccRawMed)
    AccRawMedX = AccRawMed[0]
    AccRawMedY = AccRawMed[1]
    AccRawMedZ = AccRawMed[2]

    AngRawMed=np.array([data[4],data[5],data[6]])
   # scaler = StandardScaler()
    AngRawMed = ss.zscore(AngRawMed)
    AngRawMedX = AngRawMed[0]
    AngRawMedY = AngRawMed[1]
    AngRawMedZ = AngRawMed[2]
    MeanAccMedRow = np.mean(AccRawMed)
    VarAccMedRow = np.var(AccRawMed)
    STDAccMedRow = np.std(AccRawMed)
    SMAAcc = abs(AccRawMedX)/2+(abs(AccRawMedX)+abs(AccRawMedY))/2+(abs(AccRawMedY)+abs(AccRawMedZ))/2+abs(AccRawMedZ)/2
    SMVAcc = math.sqrt(AccRawMedX**2+AccRawMedY**2+AccRawMedZ**2)
    SVHPAcc = abs(math.sqrt(AccRawMedX**2+AccRawMedY**2))
    AccRawTilt1 =(aco(AccRawMedX)).real
    AccRawSigittal = math.atan2(AccRawMedZ,AccRawMedX)
    AccMinXYZ = min(AccRawMed)
    AccMaxXYZ = max(AccRawMed)
    AccDifXYZ = AccMaxXYZ-AccMinXYZ
    IntQuartAccXYZ = iqr(AccRawMed)
    FFTrAccX = fft(record,0)
    FFTrAccY = fft(record,1)
    FFTrAccZ = fft(record,2)
    SumFFTrAcc = FFTrAccX+FFTrAccY+FFTrAccZ
    PrFFTAccX = FFTrAccX**2
    PrFFTAccY = FFTrAccY**2
    PrFFTAccZ = FFTrAccZ**2
    SumPrFFTAcc = PrFFTAccX+PrFFTAccY+PrFFTAccZ
    JerkAccX = record[2][0]-record[1][0]
    JerkAccY = record[2][1]-record[1][1]
    JerkAccZ = record[2][2]-record[1][2]
    MeanAngMedRow = np.mean(AngRawMed)
    SMVAng = math.sqrt(AngRawMedX**2+AngRawMedY**2+AngRawMedZ**2)
    AngMinXYZ = min(AngRawMed)
    AngMaxXYZ = max(AngRawMed)
    AngDifXYZ = AngMinXYZ-AngMaxXYZ
    IntQuartAngXYZ = iqr(AngRawMed)
    AngAMedianXYZ = (abs(AngRawMedX-MeanAngMedRow)+abs(AngRawMedY-MeanAngMedRow)+abs(AngRawMedZ-MeanAngMedRow))/3
    QuartX = 0.655
    QuartY = 0.322
    QuartZ = 0.212
    QuartW = 0.422
    Xfinal=[AccRawMedX,AccRawMedY,AccRawMedZ,MeanAccMedRow,VarAccMedRow,STDAccMedRow,SMAAcc,SMVAcc,SVHPAcc,AccRawTilt1,
            AccRawSigittal,AccMinXYZ, AccMaxXYZ,AccDifXYZ,IntQuartAccXYZ,FFTrAccX,FFTrAccY,FFTrAccZ,SumFFTrAcc,PrFFTAccX,
            PrFFTAccY, PrFFTAccZ,SumPrFFTAcc,JerkAccX,JerkAccY,JerkAccZ,AngRawMedX,AngRawMedY,AngRawMedZ, MeanAngMedRow,SMVAng,AngMinXYZ,AngMaxXYZ,AngDifXYZ,
            IntQuartAngXYZ,AngAMedianXYZ,QuartX,QuartY,QuartZ,QuartW]

    Xfinal=np.array(Xfinal,dtype=np.float16)

    X_test=Xfinal[DT_feature]
    X_test=X_test.reshape((1,-1))
    y_pred = DT.predict(X_test)
    print(y_pred[0])
    if y_pred[0]==1 or y_pred[0]==2:
        print("낙상이다!!")
    else:
        print('정상')
    record.pop(0)