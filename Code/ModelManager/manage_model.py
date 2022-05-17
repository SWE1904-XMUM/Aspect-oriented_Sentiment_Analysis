import pickle

# path
modelPath = '../Model/'
resultPath = '../Results/'

def save_model(model,modelName):
    pickle.dump(model, open(modelPath + modelName + '.pkl', 'wb'))

def load_model(modelName):
    model = pickle.load(open(modelPath + modelName + '.pkl', 'rb'))
    return model

def save_into_csv(result, fileName):
    result.to_csv(resultPath + fileName + '.csv', index=True)