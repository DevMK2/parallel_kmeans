import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy as dcopy
from enum import Enum


class LogType(Enum):
    Warterfall = 1
    Loop = 2


def getDataFrame(fileName):
    file = open(fileName, 'r', encoding='utf-8')
    data = pd.read_csv(file)
    file.close()

    return data


def checkLogType(df):
    type = LogType.Warterfall
    if 'iteration' in df:
        type = LogType.Loop
    return type


def addIndex(df, logType=LogType.Warterfall):
    df = dcopy(df)

    index = []

    if logType == LogType.Loop:
        for idx in df.index:
            index.append(df.message[idx] + " ("+str(df.iteration[idx])+")")
        del df['iteration']
    else:
        for idx in df.index:
            index.append(df.message[idx])

    del df['message']
    df.index=index

    return df


def addLabel(plot, logType=LogType.Warterfall):
    if logType == LogType.Loop:
        plot.xlabel('Message(iteration)');
    else:
        plot.xlabel('Message');
    plot.ylabel('Elapsed time(ms)');


def drawData(title, data, logType=LogType.Warterfall, kind='bar'):
    data.plot(kind=kind, rot=0, title=title)
    addLabel(plt, logType);


def filePostfix(path):
    return path.strip().split('.')[-1]


def drawFile(filePath):
    if filePostfix(filePath) != 'csv':
        return

    data = getDataFrame(filePath)
    if data.empty:
        return

    logType = checkLogType(data)
    data = addIndex(data, logType)

    drawData(filePath, data, logType, kind='bar')


def drawDir(dirPath):
    inDir = [os.path.join(dirPath,path) for path in os.listdir(dirPath)]

    for path in inDir:
        if os.path.isdir(path):
            drawDir(path)
        else:
            drawFile(path)


if __name__ == '__main__':
    #inputFile = "../temp/LoopResultFileFormatCorrect.csv"
    #inputFile = "../temp/ResultFileFormatCorrect.csv"
    relativeFilePath = input("파일의 상대좌표 혹은 csv파일들이 있는 디렉토리 입력:")

    if os.path.isfile(relativeFilePath):
        drawFile(relativeFilePath)
    elif os.path.isdir(relativeFilePath):
        drawDir(relativeFilePath)
    else:
        print("No such file/dir "+relativeFilePath)
        sys.exit(-1)

    plt.show();

    ##first = data.loc[:,['first']]
    ##first.plot(kind='bar');

    ##last = data.loc[:,['last']]
    ##last.plot(kind='bar');

    ##avg = data.loc[:,['average']]
    ##avg.plot(kind='bar');

    ##max = data.loc[:,['max']]
    ##max.plot(kind='bar');

    ##min = data.loc[:,['min']]
    ##min.plot(kind='bar');

