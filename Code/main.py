# import modules required
import pandas as pd
import nltk, time
from keras.models import load_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from keras.models import Sequential
from Code.DataCollection import as_scraper, gp_scraper
from Code.Preprocess import text_preprocessing, preprocess_dataframe
from Code.FeatureExtraction import feature_extraction_model
from Code.PerformanceAnalysis import analysis
from Code.AspectExtraction import extract_aspect
from Code.SentimentAnalyser import machine_learning_model, deep_learning_model
from Code.DataVisualization import visualize_data
from Code.ModelManager import manage_model

"""Path"""
datasetPath = '../Dataset/'
modelPath = '../Model/'
resultPath = '../Results/'

"""1. Data Collection"""
# Collect data with scrapper
# gp_scraper.google_play_scraper('com.google.android.youtube', 'ph','YouTube_Reviews_(gp_ph).csv')
# as_scraper.app_store_scraper('messenger', 'my','FbMessenger_Reviews_(as_my).csv')

"""2. Pre-process Dataframe & Export to csv"""
'''Manage app reviews of one app (different country)'''
# read different country app reviews
# fbAsMy = pd.read_csv(datasetPath + '/Facebook/Facebook_Reviews_(as_my).csv', nrows=200)
# fbAsPh = pd.read_csv(datasetPath + '/Facebook/Facebook_Reviews_(as_ph).csv', nrows=200)
# fbAsSg = pd.read_csv(datasetPath + '/Facebook/Facebook_Reviews_(as_sg).csv', nrows=200)
# fbGpMy = pd.read_csv(datasetPath + '/Facebook/Facebook_Reviews_(gp_my).csv', nrows=200)
# fbGpPh = pd.read_csv(datasetPath + '/Facebook/Facebook_Reviews_(gp_ph).csv', nrows=200)
# fbGpSg = pd.read_csv(datasetPath + '/Facebook/Facebook_Reviews_(gp_sg).csv', nrows=200)

# clean different country app reviews csv
# fbAsMy = preprocess_dataframe.clean_as_df(fbAsMy,'Facebook','Malaysia')
# fbAsPh = preprocess_dataframe.clean_as_df(fbAsPh,'Facebook','Philippines')
# fbAsSg = preprocess_dataframe.clean_as_df(fbAsSg,'Facebook','Singapore')
# fbGpMy = preprocess_dataframe.clean_gp_df(fbGpMy,'Facebook','Malaysia')
# fbGpPh = preprocess_dataframe.clean_gp_df(fbGpPh,'Facebook','Philippines')
# fbGpSg = preprocess_dataframe.clean_gp_df(fbGpSg,'Facebook','Singapore')

# join different country app reviews csv into one csv
# fbReview = pd.concat([fbAsMy,fbAsPh,fbAsSg,fbGpMy,fbGpPh,fbGpSg], ignore_index=True)
# fbReview.to_csv(datasetPath + 'YouTube/YouTube_Reviews.csv')

'''Manage 6 app reviews'''
# read 6 app reviews
# fb = pd.read_csv(datasetPath + 'Facebook/Facebook_Reviews.csv')
# ms = pd.read_csv(datasetPath + 'FbMessenger/Facebook_Messenger_Reviews.csv')
# ig = pd.read_csv(datasetPath + 'Instagram/Instagram_Reviews.csv')
# wc = pd.read_csv(datasetPath + 'WeChat/WeChat_Reviews.csv')
# wa = pd.read_csv(datasetPath + 'WhatsApp/WhatsApp_Reviews.csv')
# yt = pd.read_csv(datasetPath + 'YouTube/YouTube_Reviews.csv')

# clean dataframe of 6 app reviews
# fb = preprocess_dataframe.clean_df(fb)
# ms = preprocess_dataframe.clean_df(ms)
# ig = preprocess_dataframe.clean_df(ig)
# wc = preprocess_dataframe.clean_df(wc)
# wa = wa.drop(labels=['developerResponse'], axis=1)
# wa = preprocess_dataframe.clean_df(wa)
# yt = preprocess_dataframe.clean_df(yt)

# join all app reviews into one dataframe
# appReviews = pd.concat([fb,ms,ig,wc,wa,yt], ignore_index=True)

'''Manage joined app reviews (all in single csv)'''
# split app revews into sentences
# appReviews['review'] = appReviews['review'].apply(lambda x:preprocess_dataframe.split_into_sentences(x))
# appReviews = appReviews.explode('review')
# appReviews = appReviews.rename(columns={'Unnamed: 0': 'review_id','App Market Place':'app_market_place'})
# appReviews.to_csv(datasetPath + 'App_Reviews.csv')
# appReviews = appReviews.drop(labels=['Unnamed: 0'], axis=1)
# appReviews.to_csv(datasetPath + 'App_Reviews.csv')

# remove empty tokens from app reviews
# appReviews = appReviews[appReviews.clean_review != '[]']

# export app reviews dataframe into csv file
# appReviews.to_csv(datasetPath + 'App_Reviews.csv')

# read app reviews csv
# appReviews = pd.read_csv(datasetPath + 'App_Reviews.csv')

# appReviews = appReviews.drop(labels=['rating','review_title','date_time','app_name','country','app_market_place'], axis=1)
# movieReviews = movieReviews.drop(labels=['Unnamed: 0'], axis=1)
# movieReviews = movieReviews.rename(columns={'text': 'review', 'label': 'sentiment_label','clean_text':'clean_review'})

# Reviews for round 3 training
# r3 = pd.concat([movieReviews,appReviews], ignore_index=True)
# r3.to_csv(datasetPath + 'Round3_Reviews.csv')

# appReviews = appReviews.drop(labels=['Unnamed: 0','review','rating','review_title','date_time','app_name','country','app_market_place'], axis=1)
# r2Reviews = pd.concat([appReviews,movieReviews], ignore_index=True)
# r2Reviews.to_csv(datasetPath + 'Round2_Reviews.csv')

"""3. Text Pre-processing"""
# Movie reviews
# movieReviewTime = time.time()
# movieReviews['clean_text'] = movieReviews['text'].apply(lambda x: text_preprocessing.preprocess_text(x))
# analysis.time_required('pre-process time for movie reviews: ', movieReviewTime)
# movieReviews.to_csv(datasetPath + 'Movie_Reviews.csv')

# App Reviews
# appRevievTime = time.time()
# appReviews['clean_review'] = appReviews['review'].apply(lambda x: text_preprocessing.preprocess_text(x))
# analysis.time_required('pre-process time for app reviews: ',appRevievTime)
# appReviews.to_csv(datasetPath + 'App_Reviews.csv')

# test file
# t = pd.read_csv(datasetPath + 'test.csv', nrows=3)
# t['clean_text'] = t['data'].apply(lambda x: text_preprocessing.preprocess_text(x))

"""4. Assign Variable (x & y)"""
# App reviews
# appReviews = pd.read_csv(datasetPath + 'App_Reviews.csv')

# Movie reviews
# read train dataset csv file
# movieReviews = pd.read_csv(datasetPath + 'Movie_Reviews.csv')
# movieReviews = movieReviews.reset_index(drop=True)
# movieReviews = movieReviews.drop(labels=['Unnamed: 0','text'], axis=1)
# movieReviews = movieReviews.rename(columns={'clean_text': 'clean_review', 'label': 'sentiment_label'})
# xMovie = movieReviews['clean_text'].astype(str)# required if passing dataframe
# yMovie = movieReviews['label'] # for reading csv column
# xMovie = movieReviews['clean_text']

# Aspect extraction
# appReviews = pd.read_csv(datasetPath + 'App_Reviews.csv')
# xReview = appReviews['clean_review']

# Sentiment Prediction
# appReviews = pd.read_csv(datasetPath + 'App_Reviews.csv')
# xReview = appReviews['clean_review']
# yReview = appReviews['sentiment_label']

# Round 1
# appReviews = pd.read_csv(datasetPath + 'App_Reviews.csv', nrows=200)
# xReview = appReviews['clean_review']
# yReview = appReviews['sentiment_label']

# Round 2
# r2Reviews = pd.read_csv(datasetPath + 'Round2_Reviews.csv')
# xReview = r2Reviews['clean_review']
# yReview = r2Reviews['sentiment_label']

# Round 3
# r3Reviews = pd.read_csv(datasetPath + 'Round3_Reviews.csv')
# xReview = r3Reviews['clean_review']
# yReview = r3Reviews['sentiment_label']

"""5. Features Extraction"""
'''bag of words model with bigram'''
# Round 1
# bowVector1, bowVocab1 = feature_extraction_model.bow_first_round(xReview)
# manage_model.save_model(bowVector1,'Features/bow1')
# manage_model.save_model(bowVocab1,'Features/bowFea1')

# Round 2
# bowFea1 = manage_model.load_model('Features/bowFea1')
# bowVector2, bowVocab2 = feature_extraction_model.bow_retrain(xReview,bowFea1)
# manage_model.save_model(bowVector2,'Features/bow2')
# manage_model.save_model(bowVocab2,'Features/bowFea2')

# Round 3
# bowFea2 = manage_model.load_model('Features/bowFea2')
# bowVector3, bowVocab3 = feature_extraction_model.bow_retrain(xReview,bowFea2)
# manage_model.save_model(bowVector3,'Features/bow3')
# manage_model.save_model(bowVocab3,'Features/bowFea3')

'''TD-IDF model'''
# Round 1
# tfidfVector1, tfidfVocab1 = feature_extraction_model.tfidf_first_round(xReview)
# manage_model.save_model(tfidfVector1,'Features/tfidf1')
# manage_model.save_model(tfidfVocab1,'Features/tfidfFea1')

# Round 2
# tfidfFea1 = manage_model.load_model('Features/tfidfFea1')
# tfidfVector2, tfidfVocab2 = feature_extraction_model.tfidf_retrain(xReview,tfidfFea1)
# manage_model.save_model(tfidfVector2,'Features/tfidf2')
# manage_model.save_model(tfidfVocab2,'Features/tfidfFea2')

# Round 3
# tfidfFea2 = manage_model.load_model('Features/tfidfFea2')
# tfidfVector3, tfidfVocab3 = feature_extraction_model.tfidf_retrain(xReview,tfidfFea3)
# manage_model.save_model(tfidfVector3,'Features/tfidf3')
# manage_model.save_model(tfidfVocab3,'Features/tfidfFea3')

"""6. Sentiment Analyser Training"""
testSize, randomState = 0.2, False

'''Logistic Regression model'''
# Round 1 - BOW
# lr = LogisticRegression(solver='lbfgs')
# xBow1 = manage_model.load_model('Features/bow1')
# lrBowTime1 =  time.time()
# yTestLrBow1, yPredLrBow1, lrBow1 = machine_learning_model.analyzer(xBow1,yReview,testSize,randomState,lr)
# reportLrBow1, confMatrixLrBow1, accScoreLrBow1 = analysis.analyze_algorithm_performance(yTestLrBow1, yPredLrBow1,'ClassificationReport/crLrBowTt1','ConfusionMatrix/cmLrBowTt1','Accuracy/accLrBowTt1')
# minLrBow1, secLrBow1 = analysis.time_required('BOW LR, round 1 ', lrBowTime1)
# manage_model.save_model(lrBow1,'Trained/lrBowTt1')
# manage_model.save_model(minLrBow1,'Time/lrBowTtMin1')
# manage_model.save_model(secLrBow1,'Time/lrBowTtSec1')

# Round 2 - BOW
# lrBow1 = manage_model.load_model('Trained/lrBowTt1')
# xBow2 = manage_model.load_model('Features/bow2')
# lrBowTime2 =  time.time()
# yTestLrBow2, yPredLrBow2, lrBow2 = machine_learning_model.analyzer(xBow2,yReview,testSize,randomState,lrBow1)
# reportLrBow2, confMatrixLrBow2, accScoreLrBow2 = analysis.analyze_algorithm_performance(yTestLrBow2, yPredLrBow2,'ClassificationReport/crLrBowTt2','ConfusionMatrix/cmLrBowTt2','Accuracy/accLrBowTt2')
# minLrBow2, secLrBow2 = analysis.time_required('BOW LR, round 2 ', lrBowTime2)
# manage_model.save_model(lrBow2,'Trained/lrBowTt2')
# manage_model.save_model(minLrBow2,'lrBowTtMin2')
# manage_model.save_model(secLrBow2,'lrBowTtSec2')

# Round 3 - BOW
# lrBow2 = manage_model.load_model('Trained/lrBowTt2')
# xBow3 = manage_model.load_model('Features/bow3')
# lrBowTime3 =  time.time()
# yTestLrBow3, yPredLrBow3, lrBow3 = machine_learning_model.analyzer(xBow3,yReview,testSize,randomState,lrBow2)
# reportLrBow3, confMatrixLrBow3, accScoreLrBow3 = analysis.analyze_algorithm_performance(yTestLrBow3, yPredLrBow3,'ClassificationReport/crLrBowTt3','ConfusionMatrix/cmLrBowTt3','Accuracy/accLrBowTt3')
# minLrBow3, secLrBow3 = analysis.time_required('BOW LR, round 3 ', lrBowTime3)
# manage_model.save_model(lrBow3,'Trained/lrBowTt3')
# manage_model.save_model(minLrBow3,'lrBowTtMin3')
# manage_model.save_model(secLrBow3,'lrBowTtSec3')

# Round 1 - TF-IDF
# lr = LogisticRegression(solver='lbfgs')
# xtfidf1 = manage_model.load_model('Features/tfidf1')
# lrTfidfTime1 =  time.time()
# yTestLrtfidf1, yPredLrtfidf1, lrTfidf1 = machine_learning_model.analyzer(xtfidf1,yReview,testSize,randomState,lr)
# reportLrtfidf1, confMatrixLrtfidf1, accScoreLrtfidf1 = analysis.analyze_algorithm_performance(yTestLrtfidf1, yPredLrtfidf1,'ClassificationReport/crLrTfidfTt1','ConfusionMatrix/cmLrTfidfTt1','Accuracy/accLrTfidfTt1')
# minLrtfidf1, secLrtfidf1 = analysis.time_required('TF-IDF LR, round 1 ', lrTfidfTime1)
# manage_model.save_model(lrTfidf1,'Trained/lrTfidfTt1')
# manage_model.save_model(minLrtfidf1,'Time/lrTfidfTtMin1')
# manage_model.save_model(secLrtfidf1,'Time/lrTfidfTtSec1')

# Round 2 -  TF-IDF
# lrTfidf1 = manage_model.load_model('Trained/lrTfidfTt1')
# xtfidf2 = manage_model.load_model('Features/tfidf2')
# lrTfidfTime2 =  time.time()
# yTestLrtfidf2, yPredLrtfidf2, lrTfidf2 = machine_learning_model.analyzer(xtfidf2,yReview,testSize,randomState,lrTfidf1)
# reportLrtfidf2, confMatrixLrtfidf2, accScoreLrtfidf2 = analysis.analyze_algorithm_performance(yTestLrtfidf2, yPredLrtfidf2,'ClassificationReport/crLrTfidfTt2','ConfusionMatrix/cmLrTfidfTt2','Accuracy/accLrTfidfTt2')
# minLrtfidf2, secLrtfidf2 = analysis.time_required('TF-IDF LR, round 2 ', lrTfidfTime2)
# manage_model.save_model(lrTfidf2,'Trained/lrTfidfTt2')
# manage_model.save_model(minLrtfidf2,'Time/lrTfidfTtMin2')
# manage_model.save_model(secLrtfidf2,'Time/lrTfidfTtSec2')

# Round 3 -  TF-IDF
# lrTfidf2 = manage_model.load_model('Trained/lrTfidfTt2')
# xtfidf3 = manage_model.load_model('Features/tfidf3')
# lrTfidfTime3 =  time.time()
# yTestLrtfidf3, yPredLrtfidf3, lrTfidf3 = machine_learning_model.analyzer(xtfidf3,yReview,testSize,randomState,lrTfidf2)
# reportLrtfidf3, confMatrixLrtfidf3, accScoreLrtfidf3 = analysis.analyze_algorithm_performance(yTestLrtfidf3, yPredLrtfidf3,'ClassificationReport/crLrTfidfTt3','ConfusionMatrix/cmLrTfidfTt3','Accuracy/accLrTfidfTt3')
# minLrtfidf3, secLrtfidf3 = analysis.time_required('TF-IDF LR, round 3 ', lrTfidfTime3)
# manage_model.save_model(lrTfidf3,'Trained/lrTfidfTt3')
# manage_model.save_model(minLrtfidf3,'Time/lrTfidfTtMin3')
# manage_model.save_model(secLrtfidf3,'Time/lrTfidfTtSec3')

'''Multinomial Naive Bayes model'''
# Round 1 - BOW
# nb = MultinomialNB()
# xBow1 = manage_model.load_model('Features/bow1')
# nbBowTime1 =  time.time()
# yTestNbBow1, yPredNbBow1, nbBow1 = machine_learning_model.analyzer(xBow1,yReview,testSize,randomState,nb)
# reportNbBow1, confMatrixNbBow1, accScoreNbBow1 = analysis.analyze_algorithm_performance(yTestNbBow1, yPredNbBow1,'ClassificationReport/crNbBowTt1','ClassificationReport/cmNbBowTt1','ClassificationReport/accNbBowTt1')
# minNbBow1, secNbBow1 = analysis.time_required('BOW NB, round 1 ', nbBowTime1)
# manage_model.save_model(nbBow1,'Trained/nbBowTt1')
# manage_model.save_model(minNbBow1,'nbBowMin1')
# manage_model.save_model(secNbBow1,'nbBowSec1')

# Round 2 - BOW
# nbBow1 = manage_model.load_model('Trained/nbBowTt1')
# xBow2 = manage_model.load_model('Features/bow2')
# nbBowTime2 =  time.time()
# yTestNbBow2, yPredNbBow2, nbBow2 = machine_learning_model.analyzer(xBow2,yReview,testSize,randomState,nbBow1)
# reportNbBow2, confMatrixNbBow2, accScoreNbBow2 = analysis.analyze_algorithm_performance(yTestNbBow2, yPredNbBow2,'ClassificationReport/crNbBowTt2','ClassificationReport/cmNbBowTt2','ClassificationReport/accNbBowTt2')
# minNbBow2, secNbBow2 = analysis.time_required('BOW NB, round 2 ', nbBowTime2)
# manage_model.save_model(nbBow2,'Trained/nbBowTt2')
# manage_model.save_model(minNbBow2,'nbBowMin2')
# manage_model.save_model(secNbBow2,'nbBowSec2')

# Round 3 - BOW
# nbBow2 = manage_model.load_model('Trained/nbBowTt2')
# xBow3 = manage_model.load_model('Features/bow3')
# nbBowTime3 =  time.time()
# yTestNbBow3, yPredNbBow3, nbBow3 = machine_learning_model.analyzer(xBow3,yReview,testSize,randomState,nbBow2)
# reportNbBow3, confMatrixNbBow3, accScoreNbBow3 = analysis.analyze_algorithm_performance(yTestNbBow3, yPredNbBow3,'ClassificationReport/crNbBowTt3','ClassificationReport/cmNbBowTt3','ClassificationReport/accNbBowTt3')
# minNbBow3, secNbBow3 = analysis.time_required('BOW NB, round 3 ', nbBowTime3)
# manage_model.save_model(nbBow3,'Trained/nbBowTt3')
# manage_model.save_model(minNbBow3,'nbBowMin3')
# manage_model.save_model(secNbBow3,'nbBowSec3')

# Round 1 - TF-IDF
# nb = MultinomialNB()
# xTfidf1 = manage_model.load_model('Features/tfidf1')
# nbTfidfTime1 =  time.time()
# yTestNbTfidf1, yPredNbTfidf1, nbTfidf1 = machine_learning_model.analyzer(xTfidf1,yReview,testSize,randomState,nb)
# reportNbTfidf1, confMatrixNbTfidf1, accScoreNbTfidf1 = analysis.analyze_algorithm_performance(yTestNbTfidf1, yPredNbTfidf1,'ClassificationReport/crNbTfidfTt1','ClassificationReport/cmNbTfidfTt1','ClassificationReport/accNbTfidfTt1')
# minNbTfidf1, secNbTfidf1 = analysis.time_required('TF-IDF NB, round 1 ', nbTfidfTime1)
# manage_model.save_model(nbTfidf1,'Trained/nbTfidfTt1')
# manage_model.save_model(minNbTfidf1,'nbTfidfMin1')
# manage_model.save_model(secNbTfidf1,'nbTfidfSec1')

# Round 2 - BOW
# nbTfidf1 = manage_model.load_model('Trained/nbBowTt1')
# nb = MultinomialNB()
# xTfidf2 = manage_model.load_model('Features/tfidf2')
# nbTfidfTime2 =  time.time()
# yTestNbTfidf2, yPredNbTfidf2, nbTfidf2 = machine_learning_model.analyzer(xTfidf2,yReview,testSize,randomState,nbTfidf1)
# reportNbTfidf2, confMatrixNbTfidf2, accScoreNbTfidf2 = analysis.analyze_algorithm_performance(yTestNbTfidf2, yPredNbTfidf2,'ClassificationReport/crNbTfidfTt2','ClassificationReport/cmNbTfidfTt2','ClassificationReport/accNbTfidfTt2')
# minNbTfidf2, secNbTfidf2 = analysis.time_required('TF-IDF NB, round 2 ', nbTfidfTime2)
# manage_model.save_model(nbTfidf2,'Trained/nbTfidfTt2')
# manage_model.save_model(minNbTfidf2,'nbTfidfMin2')
# manage_model.save_model(secNbTfidf2,'nbTfidfSec2')

# Round 3 - BOW
# nbTfidf2 = manage_model.load_model('Trained/nbBowTt2')
# nb = MultinomialNB()
# xTfidf3 = manage_model.load_model('Features/tfidf3')
# nbTfidfTime3 =  time.time()
# yTestNbTfidf3, yPredNbTfidf3, nbTfidf3 = machine_learning_model.analyzer(xTfidf3,yReview,testSize,randomState,nbTfidf2)
# reportNbTfidf3, confMatrixNbTfidf3, accScoreNbTfidf3 = analysis.analyze_algorithm_performance(yTestNbTfidf3, yPredNbTfidf3,'ClassificationReport/crNbTfidfTt3','ClassificationReport/cmNbTfidfTt3','ClassificationReport/accNbTfidfTt3')
# minNbTfidf3, secNbTfidf3 = analysis.time_required('TF-IDF NB, round 3 ', nbTfidfTime3)
# manage_model.save_model(nbTfidf3,'Trained/nbTfidfTt3')
# manage_model.save_model(minNbTfidf3,'nbTfidfMin3')
# manage_model.save_model(secNbTfidf3,'nbTfidfSec3')

'''SVM model'''
# Round 1 - BOW
# svm = SVC(kernel='linear', C=10.0, random_state=1)
# xBow1 = manage_model.load_model('Features/bow1')
# svmBowTime1 =  time.time()
# yTestSvmBow1, yPredSvmBow1, svmBow1 = machine_learning_model.analyzer(xBow1,yReview,testSize,randomState,svm)
# reportSvmBow1, confMatrixSvmBow1, accScoreSvmBow1 = analysis.analyze_algorithm_performance(yTestSvmBow1, yPredSvmBow1,'ClassificationReport/crSvmBowTt1','ClassificationReport/cmSvmBowTt1','ClassificationReport/accSvmBowTt1')
# minSvmBow1, secSvmBow1 = analysis.time_required('BOW SVM, round 1 ', svmBowTime1)
# manage_model.save_model(svmBow1,'Trained/svmBowTt1')
# manage_model.save_model(minSvmBow1,'svmBowMin1')
# manage_model.save_model(secSvmBow1,'svmBowSec1')

# Round 2 - BOW
# svmBow1 = manage_model.load_model('Trained/svmBowTt1')
# xBow2 = manage_model.load_model('Features/bow2')
# svmBowTime2 =  time.time()
# yTestSvmBow2, yPredSvmBow2, svmBow2 = machine_learning_model.analyzer(xBow2,yReview,testSize,randomState,svmBow1)
# reportSvmBow2, confMatrixSvmBow2, accScoreSvmBow2 = analysis.analyze_algorithm_performance(yTestSvmBow2, yPredSvmBow2,'ClassificationReport/crSvmBowTt2','ClassificationReport/cmSvmBowTt2','ClassificationReport/accSvmBowTt2')
# minSvmBow2, secSvmBow2 = analysis.time_required('BOW SVM, round 2 ', svmBowTime2)
# manage_model.save_model(svmBow2,'Trained/svmBowTt2')
# manage_model.save_model(minSvmBow2,'svmBowMin2')
# manage_model.save_model(secSvmBow2,'svmBowSec2')

# Round 3 - BOW
# svmBow2 = manage_model.load_model('Trained/svmBowTt2')
# xBow3 = manage_model.load_model('Features/bow3')
# svmBowTime3 =  time.time()
# yTestSvmBow3, yPredSvmBow3, svmBow3 = machine_learning_model.analyzer(xBow3,yReview,testSize,randomState,svmBow2)
# reportSvmBow3, confMatrixSvmBow3, accScoreSvmBow3 = analysis.analyze_algorithm_performance(yTestSvmBow3, yPredSvmBow3,'ClassificationReport/crSvmBowTt3','ClassificationReport/cmSvmBowTt3','ClassificationReport/accSvmBowTt3')
# minSvmBow3, secSvmBow3 = analysis.time_required('BOW SVM, round 3 ', svmBowTime3)
# manage_model.save_model(svmBow3,'Trained/svmBowTt3')
# manage_model.save_model(minSvmBow3,'svmBowMin3')
# manage_model.save_model(secSvmBow3,'svmBowSec3')

# Round 1 - TF-IDF
# svm = SVC(kernel='linear', C=10.0, random_state=1)
# xTfidf1 = manage_model.load_model('Features/tfidf1')
# svmTfidfTime1 =  time.time()
# yTestSvmTfidf1, yPredSvmTfidf1, svmTfidf1 = machine_learning_model.analyzer(xTfidf1,yReview,testSize,randomState,svm)
# reportSvmTfidf1, confMatrixSvmTfidf1, accScoreSvmTfidf1 = analysis.analyze_algorithm_performance(yTestSvmTfidf1, yPredSvmTfidf1,'ClassificationReport/crSvmTfidfTt1','ClassificationReport/cmSvmTfidfTt1','ClassificationReport/accSvmTfidfTt1')
# minSvmTfidf1, secSvmTfidf1 = analysis.time_required('TF-IDF SVM, round 1 ', svmTfidfTime1)
# manage_model.save_model(svmTfidf1,'Trained/svmTfidfTt1')
# manage_model.save_model(minSvmTfidf1,'svmTfidfMin1')
# manage_model.save_model(secSvmTfidf1,'svmTfidfSec1')

# Round 2 - TF-IDF
# svmTfidf1 = manage_model.load_model('Trained/svmTfidfTt1')
# xTfidf2 = manage_model.load_model('Features/tfidf2')
# svmTfidfTime2 =  time.time()
# yTestSvmTfidf2, yPredSvmTfidf2, svmTfidf2 = machine_learning_model.analyzer(xTfidf2,yReview,testSize,randomState,svmTfidf1)
# reportSvmTfidf2, confMatrixSvmTfidf2, accScoreSvmTfidf2 = analysis.analyze_algorithm_performance(yTestSvmTfidf2, yPredSvmTfidf2,'ClassificationReport/crSvmTfidfTt2','ClassificationReport/cmSvmTfidfTt2','ClassificationReport/accSvmTfidfTt2')
# minSvmTfidf2, secSvmTfidf2 = analysis.time_required('TF-IDF SVM, round 2 ', svmTfidfTime2)
# manage_model.save_model(svmTfidf2,'Trained/svmTfidfTt2')
# manage_model.save_model(minSvmTfidf2,'svmTfidfMin2')
# manage_model.save_model(secSvmTfidf2,'svmTfidfSec2')

# Round 3 - TF-IDF
# svmTfidf2 = manage_model.load_model('Trained/svmTfidfTt2')
# xTfidf3 = manage_model.load_model('Features/tfidf3')
# svmTfidfTime3 =  time.time()
# yTestSvmTfidf3, yPredSvmTfidf3, svmTfidf3 = machine_learning_model.analyzer(xTfidf3,yReview,testSize,randomState,svmTfidf2)
# reportSvmTfidf3, confMatrixSvmTfidf3, accScoreSvmTfidf3 = analysis.analyze_algorithm_performance(yTestSvmTfidf3, yPredSvmTfidf3,'ClassificationReport/crSvmTfidfTt3','ClassificationReport/cmSvmTfidfTt3','ClassificationReport/accSvmTfidfTt3')
# minSvmTfidf3, secSvmTfidf3 = analysis.time_required('TF-IDF SVM, round 3 ', svmTfidfTime3)
# manage_model.save_model(svmTfidf3,'Trained/svmTfidfTt3')
# manage_model.save_model(minSvmTfidf3,'svmTfidfMin3')
# manage_model.save_model(secSvmTfidf3,'svmTfidfSec3')

'''LSTM model'''
dimension = 70

# test file
# t = pd.read_csv(datasetPath + 'test.csv', nrows=1)
# xReview = t['data']
# yReview = t['label']

# Round 1
# lstm = Sequential()
# xW2v1, yW2v1, vocabSizeW2v1, embeddingMatrixW2v1, w2v1 = feature_extraction_model.word2vec_first_round(xReview,yReview,dimension)
# lstmTime1 =  time.time()
# yTestLstm1, yPredLstm1, scoreLstm1, accuracyLstm1, summaryLstm1, lstm1 = deep_learning_model.lstm_model(xW2v1,yW2v1,vocabSizeW2v1,embeddingMatrixW2v1,dimension,testSize,randomState,lstm,'e1','lstm1','do1','d1')
# reportLstm1, confMatrixLstm1, accScoreLstm1 = analysis.analyze_algorithm_performance(yTestLstm1, yPredLstm1,'crLstmTt1','cmLstmTt1','accLstmTt1')
# minLstm1, secLstm1 = analysis.time_required('LSTM, round 1 ', lstmTime1)
# manage_model.save_model(w2v1,'w2v1')
# manage_model.save_model(minLstm1,'lstmTtMin1')
# manage_model.save_model(secLstm1,'lstmTtSec1')
# manage_model.save_model(scoreLstm1,'lstmScoreTt1')
# manage_model.save_model(accuracyLstm1,'lstmAccTt1')
# manage_model.save_model(summaryLstm1,'lstmSumTt1')
# lstm1.save('../Model/lstm1.h5')

# Round 2
# w2v1 = manage_model.load_model('Features/w2v1')
# lstm1 = load_model('../Model/Trained/lstm1.h5',compile = False)
# xW2v2, yW2v2, vocabSizeW2v2, embeddingMatrixW2v2, w2v2 = feature_extraction_model.word2vec_retrain(xReview,yReview,dimension,w2v1)
# lstmTime2 =  time.time()
# yTestLstm2, yPredLstm2, scoreLstm2, accuracyLstm2, summaryLstm2, lstm2 = deep_learning_model.lstm_model(xW2v2,yW2v2,vocabSizeW2v2,embeddingMatrixW2v2,dimension,testSize,randomState,lstm1,'e2','lstm2','do2','d2')
# reportLstm2, confMatrixLstm2, accScoreLstm2 = analysis.analyze_algorithm_performance(yTestLstm2, yPredLstm2,'crLstmTt2','cmLstmTt2','accLstmTt2')
# minLstm2, secLstm2 = analysis.time_required('LSTM, round 2 ', lstmTime2)
# manage_model.save_model(w2v2,'w2v2')
# manage_model.save_model(minLstm2,'lstmTtMin2')
# manage_model.save_model(secLstm2,'lstmTtSec2')
# manage_model.save_model(scoreLstm2,'lstmScoreTt2')
# manage_model.save_model(accuracyLstm2,'lstmAccTt2')
# manage_model.save_model(summaryLstm2,'lstmSumTt2')
# lstm2.save('../Model/lstm2.h5')

# Round 3
# w2v2 = manage_model.load_model('Features/w2v2')
# lstm2 = load_model('../Model/Trained/lstm2.h5',compile = False)
# xW2v3, yW2v3, vocabSizeW2v3, embeddingMatrixW2v3, w2v3 = feature_extraction_model.word2vec_retrain(xReview,yReview,dimension,w2v2)
# lstmTime3 =  time.time()
# yTestLstm3, yPredLstm3, scoreLstm3, accuracyLstm3, summaryLstm3, lstm3 = deep_learning_model.lstm_model(xW2v3,yW2v3,vocabSizeW2v3,embeddingMatrixW2v3,dimension,testSize,randomState,lstm2,'e3','lstm3','do3','d3')
# reportLstm3, confMatrixLstm3, accScoreLstm3 = analysis.analyze_algorithm_performance(yTestLstm3, yPredLstm3,'crLstmTt3','cmLstmTt3','accLstmTt3')
# minLstm3, secLstm3 = analysis.time_required('LSTM, round 3 ', lstmTime3)
# manage_model.save_model(w2v3,'w2v3')
# manage_model.save_model(minLstm3,'lstmTtMin3')
# manage_model.save_model(secLstm3,'lstmTtSec3')
# manage_model.save_model(scoreLstm3,'lstmScoreTt3')
# manage_model.save_model(accuracyLstm3,'lstmAccTt3')
# manage_model.save_model(summaryLstm3,'lstmSumTt3')
# lstm3.save('../Model/lstm3.h5')

"""7. Aspect Extraction"""
# Number of Topics
numOfTopic = 100
numOfWord = 10

# simple text pre-processing (only remain words that could form topics)
# appReviews = appReviews.drop(labels=['clean_review','sentiment_label'], axis=1)
# appReviews['clean_review_aspect'] = appReviews['review'].map(text_preprocessing.preprocess)

'''LDA model'''
# topicsBow, topicsTfidf = extract_aspect.lda_model(appReviews['clean_review_aspect'],numOfTopic,numOfWord)
# topicsBowPd = pd.DataFrame(topicsBow)
# topicsTfidfPd = pd.DataFrame(topicsTfidf)
# manage_model.save_model(topicsBowPd,'Topics/ldaBowDiscover')
# manage_model.save_model(topicsTfidfPd,'Topics/ldaTfidfDiscover')
# manage_model.save_into_csv(topicsBowPd,'Topics/ldaBowDiscover')
# manage_model.save_into_csv(topicsTfidfPd,'Topics/ldaTfidfDiscover')

'''wordcloud'''
# tfidf = manage_model.load_model('Aspects/tfidf')
# vector = manage_model.load_model('Aspects/appReviewsVecWordcloud')
# words = manage_model.load_model('Topics/wordcloudDiscover')
# wordcloud, words = extract_aspect.generate_wordcloud(100,tfidf,vector)
# print(words)
# manage_model.save_model(words,'Topics/wordcloudDiscover')
# visualize_data.draw_word_cloud(wordcloud)

'''BERT model'''
# bert, words, topics = extract_aspect.bert_model(xReview,numOfTopic)
# manage_model.save_model(topics,'Topics/berTopicDiscover')
# manage_model.save_model(words,'Topics/bertWordDiscover')
# berTopic = pd.DataFrame(topics)
# berWord = pd.DataFrame(words)
# berTopic.to_csv(resultPath + 'Topics/berTopicDiscover.csv')
# berWord.to_csv(resultPath + 'Topics/bertWordDiscover.csv')
# bert.visualize_topics()
# bert.visualize_barchart()

'''CorEx model'''
# x = manage_model.load_model('Aspects/appReviewsVec')
# tfidf = manage_model.load_model('Aspects/tfidf')
# corex, topics = extract_aspect.corex_model(x,numOfTopic,tfidf)
# manage_model.save_model(topics,'Topics/corexDiscover')
# corexTopic = pd.DataFrame(topics)
# corexTopic.to_csv(resultPath + 'Topics/corexDiscover.csv')
# visualize_data.visualize_corex(corex)

'''Train CorEx model'''
# topicCsv = pd.read_csv(datasetPath + 'Topics for Training.csv')
# termList = topicCsv['Terms'].map(text_preprocessing.preprocess)
# x = manage_model.load_model('Aspects/appReviewsVec')
# tfidf = manage_model.load_model('Aspects/tfidf')
# corex, topics = extract_aspect.train_corex_model(x,numOfTopic,tfidf,termList)
# corexTopic = pd.DataFrame(topics)
# corexTopic.to_csv(resultPath + 'Topics/corexTrained.csv')
# manage_model.save_model(corex, 'Topics/corexTrained')

"""8. Predicting Sentiment"""
# tfidf = manage_model.load_model('Features/tfidfFea3')
# vector, m = feature_extraction_model.tf_idf_model(xReview,'appFea',tfidf)
# vector = manage_model.load_model('Final Prediction/Sentiment/appVec')
# svmTfidf = manage_model.load_model('Trained/svmTfidfTt3')
# predSent = svmTfidf.predict(vector)
# report, confMatrix, accScore = analysis.analyze_algorithm_performance(yReview,predSent,'appReviewCr','appReviewCm','appReviewAcc')
# manage_model.save_model(predSent,'Final Prediction/Sentiment/predictedSentiment')
# pred = pd.DataFrame(predSent)
# manage_model.save_into_csv(pred, 'Final Prediction/Sentiment/predictedSentiment')

"""9. Predicting Aspects"""
# x = manage_model.load_model('Aspects/appReviewsVec')
# corexModel = manage_model.load_model('Topics/corexTrained')
# corexPredict = extract_aspect.corex_predict(x,corexModel)
# manage_model.save_model(corexPredict,'Final Prediction/Topics/predictedTopics')
# corexPredictPd = pd.DataFrame(corexPredict)
# corexPredictPd.replace({False: 0, True: 1}, inplace=True)
# manage_model.save_into_csv(corexPredictPd,'Final Prediction/Topics/predictedTopics')

"""10. Integration of Analysed Data"""
# appReviews = pd.read_csv(datasetPath + 'App_Reviews.csv')
# predictedSent = pd.read_csv(resultPath + 'Final Prediction/Sentiment/predictedSentiment.csv')
# predictedTopic = pd.read_csv(resultPath + 'Final Prediction/Topics/predictedTopics.csv')
# appReviews = appReviews.drop(labels=['sentiment_label', 'clean_review'], axis=1)
# predictedSent = predictedSent.drop(labels=['Unnamed: 0'], axis=1)
# predictedSent = predictedSent.rename(columns={'0': 'predicted_sentiment'})
# predictedTopic = predictedTopic.drop(labels=['Unnamed: 0'], axis=1)
# predictedTopic = predictedTopic.rename(
#     columns={'0': 'video', '1': 'like', '2': 'bug', '3': 'photo', '4': 'theme', '5': 'bias', '6': 'version',
#              '7': 'font', '8': 'screen_smoothess', '9': 'conversation', '10': 'user_support', '11': 'hotspot',
#              '12': 'connection', '13': 'search_function', '14': 'advertisement', '15': 'voting_function',
#              '16': 'voice/video_call', '17': 'notification', '18': 'scam', '19': 'post', '20': 'media',
#              '21': 'tracking_algorithm', '22': 'account', '23': 'entertainment', '24': 'security', '25': 'dating',
#              '26': 'dislike', '27': 'audio', '28': 'storage', '29': 'sticker', '30': 'battery', '31': 'game'})
# predictedAppReviews = pd.concat([appReviews, predictedSent, predictedTopic], axis=1)
# predictedAppReviews.to_csv(datasetPath + 'Predicted_App_Reviews.csv')
