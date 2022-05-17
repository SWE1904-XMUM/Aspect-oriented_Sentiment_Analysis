import pandas as pd
import numpy as np
import string, emoji, re, bs4
from nltk.corpus import stopwords, words, wordnet
from nltk import word_tokenize, pos_tag
from autocorrect import Speller
from nltk.stem import PorterStemmer
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

class AntonymReplacer(object):
    def replace(self,word,pos=None):
        antonyms=set()
        for syn in wordnet.synsets(word,pos=pos):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms.add(antonym.name())
        if len(antonyms)==1:
            return antonyms.pop()
        else:
            return None
    def replace_negations(self,sent):
        i,l=0,len(sent)
        words=[]
        while i<l:
            word=sent[i]
            if word.lower()=='not' and i+1<l:
                ant=self.replace(sent[i+1])
                if ant:
                    words.append(ant)
                    i+=2
                    continue
            words.append(word)
            i+=1
        return words

def preprocess_text(text):
    # 1. lowercasing text
    text = text.lower()

    # 2. convert emoji into written text
    text = emoji.demojize(text)

    # 3. remove html tags
    text = bs4.BeautifulSoup(text, features='html.parser').text

    # 4. remove punctuations
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ')

    # 5. remove digits
    text = re.sub(r'[0-9]', '', text)

    # 6. tokenization
    text = word_tokenize(text)
    # remove empty tokens
    text = [t for t in text if len(t) > 0]

    # 7. replace negations with antonyms
    replacer = AntonymReplacer()
    text = replacer.replace_negations(text)

    # 8. remove stopwords (in english)
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]

    # 9. corrected misspelled words
    spellChecker = Speller(lang='en')
    text = [spellChecker(x) for x in text]

    # 10. stemming
    ps = PorterStemmer()
    text = [w.replace(w,ps.stem(w)) for w in text]

    return text

'''simple text pre-processing for aspect extraction'''
def lemmatize_stemming(text):
    stemmer = SnowballStemmer('english')
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result