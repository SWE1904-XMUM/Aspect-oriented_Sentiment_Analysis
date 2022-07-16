from sklearn.model_selection import train_test_split, KFold, cross_val_score
from Code.PerformanceAnalysis import analysis
from Code.ModelManager import manage_model
import pandas as pd

def analyzer(x,y,testSize,randomState,model):
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=testSize, random_state=randomState)
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    return yTest, yPred, model