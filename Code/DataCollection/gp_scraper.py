from google_play_scraper import Sort, reviews_all, reviews
import pandas as pd
import numpy as np

def google_play_scraper(appName,country,csvFileName):
    r, continuation_token = reviews(
        appName,
        lang='en',  # defaults to 'en'
        country=country,  # defaults to 'us'
        sort=Sort.NEWEST,  # defaults to Sort.MOST_RELEVANT
        count=500,  # defaults to 100
    )

    df = pd.DataFrame(np.array(r), columns=['review'])
    df = df.join(pd.DataFrame(df.pop('review').tolist()))
    df.head()
    df.to_csv('C:/Users/user/Desktop/Software Engineering/FYP/FYP Code/Dataset/' + csvFileName)