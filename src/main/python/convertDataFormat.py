import pandas as pd
import arff
import glob
import os

inputPath = '/media/Isabunbun shared folder/TSC-Share/data/ORGSplited'
outputPath =  '../java/experiments/data/tsc/generated'
simulators = os.listdir(inputPath)

for simulator in simulators:
    for f in os.listdir("{}/{}".format(inputPath, simulator)):
        meta = f.split('_')
        signalStrength = meta[0][1:]
        sampleNum = meta[1][1:]
        sampleLength = meta[2][1:]
        trainTest = meta[3][:-5]
        newName = "{}_S{}_N{}_L{}".format(simulator, str(signalStrength).replace(".", ""), sampleNum, sampleLength)
        savePath = '{}/{}/{}_{}.arff'.format(outputPath, newName, newName, trainTest.upper())
        if (meta[3][-5] == "y") or os.path.exists(savePath) :
            continue
        print(simulator, meta)
        dfX = pd.read_csv("{}/{}/{}".format(inputPath, simulator, f), index_col = 0).reset_index(drop = True)
        dfy = pd.read_csv( "{}/{}/{}y.csv".format(inputPath, simulator, f[:-5]), index_col = 0).reset_index(drop = True)
        dfy['0'] = dfy['0'].astype(bool)
        dfXy = pd.concat([dfX,dfy['0']], axis = 1)

        try:
            os.mkdir('{}/{}/'.format(outputPath, newName))
        except:
            pass
        arff.dump( savePath, dfXy.values, relation = newName )
