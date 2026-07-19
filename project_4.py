##### Python codes

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

 # Prevents Streamlit from overwriting the Plotly color palette.
import plotly.io as pio
pio.templates.default = 'plotly'

# Page settings
st.set_page_config(
    page_title='Project 4', 
    page_icon='📊', 
    layout='wide'
)

# Project Decription
st.header('Project 4')
st.markdown('''
    **Interactive 3D visualization of correlated assets, using clustering and dimensionality reduction.**
''')

# Function to collect data and calculate percentage returns
def get_returns(tickers, start_date, end_date):
    stock_data = []

    # Percorre cada ticker e coleta sua cotação # 
    for ticker in tickers:
        data = yf.Ticker(ticker).history(start=start_date , end=end_date)
        data['ticker'] = ticker
        data.index = pd.to_datetime(data.index).date
        stock_data.append(data)

    # Iterates through each ticker and collects its quote.
    stock_data = pd.concat(stock_data, axis=0)[['Close', 'ticker']].fillna(0)

    # Calcula os retornos em porcentagem para todas ações
    vec_stocks = []
    for ticker in tickers:
        df = stock_data[stock_data['ticker'] == ticker]
        vec_stocks.append(df['Close'].pct_change().dropna())

    vec_stocks = pd.DataFrame(vec_stocks, index=tickers).fillna(0)
    vec_stocks.index.name = 'Ticker'
    return vec_stocks

# Side Bar settings
st.sidebar.header('Options Menu')

tickers = st.sidebar.text_area(
    'Enter the tickers separated by commas', 
    value='MSFT, TSLA, NVDA, CVCB3.SA, PETR3.SA, PETR4.SA, ^BVSP, ^SPX, BTC-USD'
)

# Definition of the start date (5 years back) and the end date
default_start = pd.to_datetime('today') - pd.DateOffset(days=365 * 5)
default_end = pd.to_datetime('today')
start_date = st.sidebar.date_input('Data inicial', value=default_start)
end_date = st.sidebar.date_input('Data final', value=default_end)

# Tickers' list
tickers_list = [ticker.strip() for ticker in tickers.split(',') if ticker.strip()]

# Sets the maximum number of clusters based on the number of tickers.
num_clusters = st.sidebar.slider(
    'Number of Clusters', 
    min_value=2, 
    max_value=len(tickers_list), 
    value=3
)

# Calculates percentage returns
if len(tickers_list) > 3:
    vec_stocks = get_returns(tickers_list, start_date, end_date)

    # Clusters the assets
    cluster_model = KMeans(n_clusters=num_clusters, random_state=1)
    preds = cluster_model.fit_predict(vec_stocks)

    # Apply PCA to reduce dimensionality to 3 (3D).
    pca = PCA(n_components=3)
    vec_stocks_3d = pd.DataFrame(
        pca.fit_transform(vec_stocks),
        index=tickers_list,
        columns=['x', 'y', 'z']
    )    
    vec_stocks_3d['ticker'] = vec_stocks_3d.index
    vec_stocks_3d['cluster'] = preds

    # Plots the interactive 3D scatterplot with the coordinates and cluster colors.
    fig = px.scatter_3d(
        vec_stocks_3d,
        x='x', y='y', z='z',
        color='cluster',
        text='ticker',
        hover_name='ticker'
    )
    fig.update_layout(
        width=800,
        height=600
    )
    st.plotly_chart(fig)
else:
    st.warning('Enter a set of at least 4 valid tickers, separated by commas.')

st.sidebar.markdown('''
    <p style="margin-top: 30px; text-align: center">
        Python Projects for the Financial Market
    
    </p>
''', unsafe_allow_html=True)
