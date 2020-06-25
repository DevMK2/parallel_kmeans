#-*- coding:utf-8 -*-
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy as dcopy
from enum import Enum

RESULT_PATH = "___visualize_results___"
ALERT = "visualize results directory has already been, it can overwrite previous results.\n Do you want to continue?(!y|n)"

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


def saveCurrFig(fileName):
    plt.savefig(os.path.join(RESULT_PATH, fileName));


def drawData(title, data, logType=LogType.Warterfall, kind='bar'):
    data.plot(kind=kind, rot=0, title=title)
    addLabel(plt, logType);
    saveCurrFig(title.replace('.csv','')+'.png')


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
    if not os.path.isdir(RESULT_PATH):
        os.mkdir(RESULT_PATH)
    elif input(ALERT) in ('n','N'):
        exit(0)

    relativeFilePath = input("\n Plz type relative path of csv file or dir path :")

    if os.path.isfile(relativeFilePath):
        drawFile(relativeFilePath)
    elif os.path.isdir(relativeFilePath):
        drawDir(relativeFilePath)
    else:
        print("No such file/dir "+relativeFilePath)
        sys.exit(-1)

    plt.show();
    print(relativeFilePath)
