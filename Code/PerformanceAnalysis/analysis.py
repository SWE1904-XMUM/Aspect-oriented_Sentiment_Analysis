import time
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from Code.ModelManager import manage_model

# analyze performance of algorithm
def analyze_algorithm_performance(yTest,yPred,reportName,cmName,accName):
    report = pd.DataFrame(classification_report(yTest, yPred,target_names=['Negative','Positive'],output_dict=True)).transpose()
    confMatrix = confusion_matrix(y_true=yTest, y_pred=yPred)
    confMatrixPd = pd.DataFrame(confMatrix)
    accScore = accuracy_score(yTest, yPred)

    manage_model.save_into_csv(report, reportName)
    manage_model.save_model(report, reportName)
    manage_model.save_into_csv(confMatrixPd, cmName)
    manage_model.save_model(confMatrix, cmName)
    manage_model.save_model(accScore, accName)

    return report, confMatrix, accScore

# running time
def time_required(message,t):
    min = (time.time() - t) / 60
    sec = (time.time() - t)
    print('Time - ' + message + ': {} mins or {} sec'.format(round(min, 6),sec))
    return min, sec