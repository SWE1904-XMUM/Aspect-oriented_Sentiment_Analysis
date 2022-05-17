from app_store_scraper import AppStore
import pandas as pd
import numpy as np

def app_store_scraper(appName,country,csvFileName):
    app = AppStore(country=country, app_name=appName)
    app.review(how_many=500)

    data = pd.DataFrame(np.array(app.reviews), columns=['review'])
    df = data.join(pd.DataFrame(data.pop('review').tolist()))
    df.head()
    df.to_csv('C:/Users/user/Desktop/Software Engineering/FYP/FYP Code/Dataset/' + csvFileName)