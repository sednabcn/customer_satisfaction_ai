import sqlite3
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import string
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats
import chart_studio
import chart_studio.plotly as py
from plotly.tools import FigureFactory as FF
from plotly.graph_objects import Bar, Scatter, Marker, Layout, Choropleth, Histogram
from IPython.display import display
import matplotlib.pyplot as plt
chart_studio.tools.set_credentials_file(username='sednachar', api_key='tjPVjxbqvP6QNEkMMmVW')
filename='airlinessat.csv'
display(pd.read_csv(filename,nrows=5).head())
conn= sqlite3.connect(':memory:')
db_conn = create_engine("sqlite+pysqlite:///:memory:", echo=True)

chunks=5000
for data in pd.read_csv(filename, chunksize=chunks, iterator=True, encoding='utf-8'):
    data = data.rename(columns={col:col.replace('_',' ') for col in data.columns})
    data = data.rename(columns={col:col.strip() for col in data.columns})
    data = data.rename(columns={col:string.capwords(col) for col in data.columns})
    data = data.rename(columns={col:col.replace(' ','') for col in data.columns})
    data.to_sql('data',db_conn, if_exists='append')

df=pd.read_sql_query('SELECT * FROM data LIMIT 10', db_conn)
print(df)

query = pd.read_sql_query('SELECT satisfaction, Age,FlightDistance, CustomerType, Class '
                         'FROM data '
                          'ORDER BY Age DESC',db_conn)
print(query)
#=======================

#=========================
# plot the dataframe
#query.plot(x='Age', y='FlightDistance', kind="bar", figsize=(9, 8))
#plt.show()
