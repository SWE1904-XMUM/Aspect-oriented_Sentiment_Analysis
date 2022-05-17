from keras.layers import Dense, Dropout, Embedding, LSTM
from sklearn.model_selection import train_test_split
from Code.FeatureExtraction import feature_extraction_model
from Code.ModelManager import manage_model
import numpy as np

# LSTM model
def lstm_model(x,y,vocabSize,embeddingMatrix,dimension,testSize,randomState,lstm,embeddingName,lstmName,dropoutName,denseName):
    # split into train & test set
    xTrain, xTest, yTrain, yTest = train_test_split(x,y,test_size=testSize,random_state=randomState)

    # add embedding layer
    embeddingLayer = Embedding(input_dim=vocabSize, output_dim=dimension, weights=[embeddingMatrix],input_length=dimension,trainable=True,name=embeddingName)
    lstm.add(embeddingLayer)
    lstm.add(Dropout(0.2,name=dropoutName))
    lstm.add(LSTM(dimension,input_shape=(dimension,128),name=lstmName))
    lstm.add(Dense(2, activation='sigmoid',name=denseName))
    lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    lstm.fit(xTrain, yTrain, batch_size=32, epochs=5, verbose=5)
    score, accuracy = lstm.evaluate(xTest,yTest,batch_size=32,verbose=2)
    yPred = lstm.predict(xTest)
    yPred = np.argmax(yPred, axis=1)
    yTest = np.argmax(yTest, axis=1)

    summary = lstm.summary()

    return yTest, yPred, score, accuracy, summary, lstm