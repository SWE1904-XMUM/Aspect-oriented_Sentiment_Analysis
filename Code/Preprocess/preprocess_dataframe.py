from nltk import tokenize

def clean_as_df(df, appName, country):
    df = df.drop(labels=['Unnamed: 0', 'isEdited', 'userName'], axis=1)
    df = df.rename(columns={'title': 'review_title', 'date': 'date_time'})
    df['app_name'] = appName
    df['country'] = country
    df['app_market_place'] = 'Apple App Store'
    return df

def clean_gp_df(df, appName, country):
    df = df.drop(labels=['Unnamed: 0', 'reviewId', 'userName', 'userImage',
                         'thumbsUpCount', 'reviewCreatedVersion',
                         'replyContent', 'repliedAt'], axis=1)
    df = df.rename(columns={'content': 'review', 'score': 'rating',
                            'at': 'date_time'})
    df['app_name'] = appName
    df['country'] = country
    df['app_market_place'] = 'Google Play Store'
    return df

def clean_df(df):
    df = df.drop(labels=['Unnamed: 0'], axis=1)
    return df

def split_into_sentences(text):
    return tokenize.sent_tokenize(text)
