import os
import pandas as pd

from os import listdir
from os.path import isfile, join

mtsSummaryDF = pd.read_csv('/media/Isabunbun shared folder/TSC-Share/summaryMultivariate.csv', index_col = 0)
classifiers = listdir("/home/isabella/Documents/TSC/tsml/results")
col = ["simModel", "simParam",  "sampleNum", "sampleLength", "classification","accuracy", "typeI", "typeII"]
resultDF = pd.DataFrame(columns = col)
# UEANaming = {'FaceDetection':'FaceDetection',
#              'SelfRegulationSCP1': 'SelfRegulationSCP1',
#              'SelfRegulationsSCP2':'SelfRegulationSCP2',
#              'FingerMovement':'FingerMovements',
#              'MotorImagery':'MotorImagery',
#              'Heartbeat':'Heartbeat',
#              'FingerMovements':'FingerMovements'}

for classifier in classifiers:
    resultDir = f"/home/isabella/Documents/TSC/tsml/results/{classifier}/Predictions/"
    dataSets =  listdir(resultDir)
    print(classifier)
    print(dataSets)
    for ds in dataSets:
        if ds == "null":
            continue
        split = ds.split('_')
        if len(split) == 1:
            simModel = "UEA"
            simParam = split[0]
            simNum = mtsSummaryDF.loc[simParam, 'TrainSize'] + mtsSummaryDF.loc[simParam, 'TestSize']
            simLength = mtsSummaryDF.loc[simParam, 'SeriesLength']
            simDim = mtsSummaryDF.loc[simParam, 'NumDimensions']
        else:
            simModel = split[0]
            simParam = split[1][1:].replace("0", "0.",1)
            simNum = split[2][1:]
            simLength = split[3][1:]
        ACC = 0
        TI = 0
        TII = 0
        count = 0
        path = resultDir + ds + '/'
        for fold in listdir(path):
            foldResultDF = pd.read_csv(path+fold, skiprows=[1,2])
            foldResultDF.iloc[:,0] = foldResultDF.iloc[:,0].astype(int)
            count += 1
            ACC += sum(foldResultDF.iloc[:,0] == foldResultDF.iloc[:,1]) / len(foldResultDF)
            TI += sum((foldResultDF.iloc[:,0] == 0) & (foldResultDF.iloc[:,1] == 1)) / len(foldResultDF[foldResultDF.iloc[:,0] == 0])
            TII += sum((foldResultDF.iloc[:,0] == 1) & (foldResultDF.iloc[:,1] == 0)) / len(foldResultDF[foldResultDF.iloc[:,0] == 1])
        if count >= 1:
            resultDF = resultDF.append({'simModel': simModel, "simParam": simParam,"classification": classifier , "sampleNum": simNum, "sampleLength": simLength, "classification": classifier, 'accuracy': ACC/count, 'typeI': TI/count, 'typeII':TII/count} , ignore_index = True)
resultDF.to_csv("/media/Isabunbun shared folder/TSC-Share/bmClassifiers_resultSummary.csv")
