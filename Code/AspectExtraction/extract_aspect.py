from wordcloud import WordCloud
from corextopic import corextopic as ct
import corextopic.vis_topic as vt
from gensim.models.ldamodel import LdaModel
from gensim.models import TfidfModel
import scipy.sparse as ss
import numpy as np
from bertopic import BERTopic
from gensim.corpora import Dictionary

# wordcloud
def generate_wordcloud(numOfWord,tfidf,vector):
    tfidfWeights = [(word, vector.getcol(idx).sum()) for word, idx in tfidf.vocabulary_.items()]

    wordcloud = WordCloud(
        background_color='white',
        max_words=numOfWord,
        max_font_size=40,
        scale=4).fit_words(dict(tfidfWeights))
    words = wordcloud.words_
    return wordcloud, words

# LDA model
def lda_model(x,numOfTopic,numOfWord):
    dictionary = Dictionary(x)

    corpusBow = [dictionary.doc2bow(text) for text in x]
    ldaBow = LdaModel(corpusBow, num_topics=numOfTopic, id2word=dictionary, passes=20)
    topicsBow = ldaBow.print_topics(num_topics=numOfTopic, num_words=numOfWord)

    tfidf = TfidfModel(corpusBow)
    corpusTfidf = tfidf[corpusBow]
    ldaTfidf = LdaModel(corpusTfidf, num_topics=numOfTopic, id2word=dictionary, passes=20)
    topicsTfidf = ldaTfidf.print_topics(num_topics=numOfTopic, num_words=numOfWord)

    return topicsBow, topicsTfidf

# BERT model
def bert_model(x, numOfTopic):
    bert = BERTopic(language='english')
    t,prob = bert.fit_transform(x)

    # extract most frequent words
    words = bert.get_topic_info()

    # get individual topics
    topics = bert.get_topic(0)

    return bert, words, topics

# keyword extraction - CorEx
def corex_model(x,noOfTopics,vectorizer):
    x = ss.csr_matrix(x)
    words = list(np.asarray(vectorizer.get_feature_names()))
    corex = ct.Corex(n_hidden=noOfTopics, words=words, max_iter=1000, verbose=False, seed=2020)
    corex.fit(x, words=words)
    topics = corex.get_topics()
    return corex, topics

def train_corex_model(x,noOfTopics,vectorizer,termList):
    x = ss.csr_matrix(x)
    words = list(np.asarray(vectorizer.get_feature_names()))
    corex = ct.Corex(n_hidden=noOfTopics, words=words, max_iter=1000, verbose=False, seed=2020)
    corex.fit(x, words=words, anchors=termList, anchor_strength=3)
    topics = corex.get_topics()
    vt.vis_rep(corex, column_label=words, prefix='../Results/Topics/Corex')
    return corex, topics

def corex_predict(x,model):
    x = ss.csr_matrix(x)
    predict = model.predict(x)
    return predict