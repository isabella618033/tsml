import os
import pandas as pd

from os import listdir
from os.path import isfile, join

def setUpBackgroundDF():
    mtsSummaryDF = pd.read_csv('/media/Isabunbun shared folder/TSC-Share/results/metaData/summaryMultivariate.csv', index_col = 0)
    utsSummaryDF = pd.read_csv('/media/Isabunbun shared folder/TSC-Share/results/metaData/summaryUnivariate.csv', index_col = 0)
    return (mtsSummaryDF, utsSummaryDF)

def setUpClassificationParam(split, mtsSummaryDF, utsSummaryDF):
    if len(split) == 1:
        try:
            simModel = "UEA"
            simParam = split[0]
            simNum = mtsSummaryDF.loc[simParam, 'TrainSize'] + mtsSummaryDF.loc[simParam, 'TestSize']
            simLength = mtsSummaryDF.loc[simParam, 'SeriesLength']
            simDim = mtsSummaryDF.loc[simParam, 'NumDimensions']
        except KeyError:
            simModel = "benchmark"
            simParam = split[0]
            simNum = utsSummaryDF.loc[simParam, 'numTrainCases'] + utsSummaryDF.loc[simParam, 'numTestCase']
            simLength = utsSummaryDF.loc[simParam, 'seriesLength']
    else:
        simModel = split[0]
        simParam = split[1][1:].replace("0", "0.",1)
        simNum = split[2][1:]
        simLength = split[3][1:]
    return(simModel, simParam, simNum, simLength)

def fetchTITII(resultDir, ds, simParam):
    swapped = False
    ACC_sum = 0
    TI_sum = 0
    TII_sum = 0
    foldCount = 0
    path = resultDir + ds + '/'
    for fold in listdir(path):
        foldResultDF = pd.read_csv(path+fold, skiprows=[1,2])
        foldResultDF.iloc[:,0] = foldResultDF.iloc[:,0].astype(int)
        foldCount += 1
        ACC_sum += sum(foldResultDF.iloc[:,0] == foldResultDF.iloc[:,1]) / len(foldResultDF)
        TI_sum += sum((foldResultDF.iloc[:,0] == 0) & (foldResultDF.iloc[:,1] == 1)) / len(foldResultDF[foldResultDF.iloc[:,0] == 0])
        TII_sum += sum((foldResultDF.iloc[:,0] == 1) & (foldResultDF.iloc[:,1] == 0)) / len(foldResultDF[foldResultDF.iloc[:,0] == 1])
        #print(simParam, sum(foldResultDF.iloc[:,0] == 0), sum(foldResultDF.iloc[:,0] == 1))
    if foldCount >= 1:
        ACC = ACC_sum / foldCount
        TI = TI_sum / foldCount
        TII = TII_sum / foldCount
    else:
        ACC = ACC_sum
        TI = TI_sum
        TII = TII_sum

    if simParam in ['ECG200', 'Earthquakes', 'Heartbeat']:
        temp = TI
        TI = TII
        TII = temp
        swapped = True
    return (ACC, TI, TII, foldCount, swapped)

def getbmClassifierResult():
    col = ["simModel", "simParam",  "sampleNum", "sampleLength", "classification","accuracy", "typeI", "typeII"]
    resultDF = pd.DataFrame(columns = col)
    backgroundDF = setUpBackgroundDF()
    classifiers = listdir("/home/isabella/Documents/TSC/tsml/results")
    for classifier in classifiers:
        resultDir = f"/home/isabella/Documents/TSC/tsml/results/{classifier}/Predictions/"
        dataSets =  listdir(resultDir)
        for ds in dataSets:
            if ds == "null":
                continue
            split = ds.split('_')
            simModel, simParam, simNum, simLength = setUpClassificationParam(split, *backgroundDF)
            ACC, TI, TII, foldCount, swapped = fetchTITII(resultDir, ds, simParam)
            if foldCount >= 1:
                resultDF = resultDF.append({'simModel': simModel, "simParam": simParam,"classification": classifier , "sampleNum": simNum, "sampleLength": simLength, "classification": classifier, 'accuracy': ACC, 'typeI': TI, 'typeII':TII, 'fold': foldCount, 'swapped': swapped} , ignore_index = True)
    resultDF.to_csv("/media/Isabunbun shared folder/TSC-Share/results/tableSummary/results_bmClassifiers_groupedFold.csv")

if __name__ == "__main__":
    getbmClassifierResult()
