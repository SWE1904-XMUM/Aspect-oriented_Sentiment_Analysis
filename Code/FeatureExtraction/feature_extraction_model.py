from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec, KeyedVectors
from keras.preprocessing.text import Tokenizer
import gensim, multiprocessing
import numpy as np

def bow_first_round(x):
    cv = CountVectorizer(ngram_range=(1, 2))
    vector, vocab = bow_ngram_model(x, cv)
    return vector, vocab

def bow_retrain(x, model):
    cv = CountVectorizer(ngram_range=(1, 2), vocabulary=model.vocabulary_)
    vector, vocab = bow_ngram_model(x, cv)
    return vector, vocab

def tfidf_first_round(x):
    tfIdf = TfidfVectorizer(max_features=10000)
    vector, vocab = tf_idf_model(x,tfIdf)
    return vector, vocab

def tfidf_retrain(x, model):
    tfIdf = TfidfVectorizer(max_features=10000, vocabulary=model.vocabulary_)
    vector, vocab = tf_idf_model(x,tfIdf)
    return vector, vocab

def word2vec_first_round(x,y,dimension):
    cores = multiprocessing.cpu_count()
    w2v = Word2Vec(sg=1, min_count=10,  window =10, epochs = 20, workers=cores - 1)
    w2v.build_vocab(x, progress_per=10000)
    x, y, vocabSize, embeddingMatrix, model = word2vec_model(x,y,dimension,w2v)
    return x, y, vocabSize, embeddingMatrix, model

def word2vec_retrain(x,y,dimension,w2v):
    w2v.build_vocab(x, progress_per=10000, update=True)
    x, y, vocabSize, embeddingMatrix, model = word2vec_model(x,y,dimension,w2v)
    return x, y, vocabSize, embeddingMatrix, model

# bag of words model with n-grams
def bow_ngram_model(x, bow):
    vector = bow.fit_transform(x)
    vocab = bow.fit(x)
    return vector, vocab

# TF-IDF model
def tf_idf_model(x, tfIdf):
    vector = tfIdf.fit_transform(x).toarray()
    vocab = tfIdf.fit(x)
    return vector, vocab

# Word2Vec model
def word2vec_model(x,y,dimension,w2v):
    w2v.train(x, total_examples=w2v.corpus_count, epochs=20, report_delay=1)

    vocabSize = w2v.corpus_count + 1
    tokenizer = Tokenizer(num_words=20)
    tokenizer.fit_on_texts(x)
    sequence = tokenizer.sequences_to_texts(x)
    embeddingMatrix = np.zeros((vocabSize, dimension))
    x = pad_sequences(sequence, maxlen=dimension)
    y = LabelEncoder().fit_transform(y)
    y = to_categorical(np.asarray(y))

    return x, y, vocabSize, embeddingMatrix, w2v