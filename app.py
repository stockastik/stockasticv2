
#=================================================================================================================
#====================== LIBRARIES ================================================================================
#=================================================================================================================

import streamlit as st #helps in creating the webapp
import pickle
from datetime import timedelta #work with dates
from sklearn.preprocessing import RobustScaler #normalize data
import pandas as pd #dataframes
import plotly.express as px #helps draw the graphics
import flair #sentiment analysis model
import requests #makes requests for the twitter API
import ta #technical indicators
import numpy as np #numerical operations and more
import matplotlib.pyplot as plt #plotting
from keras.models import Sequential
import plotly.graph_objects as go
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from yahoo_fin.stock_info import get_data
from PIL import Image
import tkinter as tk 

#========================================================================================================================
#====================== GLOBAL VARIABLES ================================================================================
#========================================================================================================================

allowed_intervals = ("Daily", "Weekly", "Monthly")
prev_days_trend = 0

#=================================================================================================================
#====================== VARIOUS FUNCTIONS ========================================================================
#=================================================================================================================

#==================== SPLITS DATA SEQUENCES TO MAKE PREDICTIONS ==================================================

def split_sequence(seq, n_steps_in, n_steps_out):

  X, y = [], []

  for i in range(len(seq)):

    end = i + n_steps_in
    out_end = end + n_steps_out

    if out_end > len(seq):
      break
    
    seq_x, seq_y = seq[i:end, :], seq[end:out_end, 0]
    X.append(seq_x)
    y.append(seq_y)

  return np.array(X), np.array(y)

#==================== CREATES THE LAYERS OF THE MODEL ==================================================

def layer_maker(model, n_layers, n_nodes, activation, drop = None, d_rate =.5):

  for x in range(1, n_layers + 1):

    model.add(LSTM(n_nodes, activation=activation, return_sequences=True))

    try:
      if x % drop == 0:
        model.add(Dropout(d_rate))
    except:
      pass

#==================== CREATES OVERLAP OF TRUE AND PREDICTED VALUES ======================================

def validater(model, df, n_per_in, n_per_out, n_features, close_scaler):

  predictions = pd.DataFrame(index=df.index, columns=[df.columns[0]])

  for i in range(n_per_in, len(df)-n_per_in, n_per_out):

    x = df[-i-n_per_in: -i]

    yhat = model.predict(np.array(x).reshape(1, n_per_in, n_features))
    yhat = close_scaler.inverse_transform(yhat)[0]

    pred_df = pd.DataFrame(yhat, 
                           index=pd.date_range(start=x.index[-1],
                                               periods=len(yhat),
                                               freq="B"),
                          columns=[x.columns[0]])
    predictions.update(pred_df)

  return predictions

#==================== CALCULATES RMSE OF THE PREDICTIONS ================================================

def val_rmse(df1, df2):

  df = df1.copy()
  df['close2'] = df2.close

  df.dropna(inplace=True)
  df['diff'] = df.close - df.close2

  rms = (df[['diff']]**2).mean()

  return float(np.sqrt((rms)))

#=================================================================================================================
#====================== MAIN FUNCTION ============================================================================
#=================================================================================================================

def show_predict_pge():

#==================== DRAWS MAIN PAGE AND HANDLES INPUTS ===============================================

    image = Image.open('logo1.png')
    favicon = Image.open('logo_favicon.png')
    st.set_page_config(page_title='Stockastic V1.0', page_icon=favicon)
    st.image(image, caption='YOUR INVESTING BUDDY')
    st.title("WELCOME TO STOCKASTIC!")
    ticker = st.text_input("Stock to analyze (Insert the Ticker Smbol)")
    interval = st.selectbox("Interval", allowed_intervals).lower()
    if interval == "daily":
      interval2 = "1d"
    if interval == "monthly":
      interval2 = "1mo"
    if interval == "weekly":
      interval2 = "1wk"
    start_date = "2020-01-01"
    print(start_date)
    end_date = st.date_input("Start predicting from")
    n_per_out = st.slider("Number of intervals to predict", 1, 365, 1)
    n_per_in = n_per_out*3

#=================================================================================================================
#====================== RUNS ON "LET'S FO THIS BUTTON" (PREDICTIONS) =============================================
#=================================================================================================================

    if st.button("LET'S DO THIS!"):

#==================== GETS DATA ==================================================================================

      df = get_data(ticker, start_date = start_date, end_date = end_date, index_as_date = False, interval = interval2)
      df['date'] = pd.to_datetime(df.date)

#==================== DRAWS PAST DATA GRAPH =======================================================================

      fig = px.line(df, x= 'date', y='close', color_discrete_sequence=["#0EBDEE"], width = 1000, height = 600, title = 'PRICE OF THE INSTRUMENT OVER TIME')
      fig.update_xaxes(
          rangeslider_visible=True
      )
      fig.update_xaxes(title_text='Time')
      fig.update_yaxes(title_text='Price')
      st.plotly_chart(fig, use_container_width=True)

#==================== PREPROCESSES DATA ==================================================================================

      df.set_index('date', inplace=True)
      df.dropna(inplace=True) 
      df = ta.add_all_ta_features(df, open = "open", high = "high", low = "low", close = "close", volume = "volume", fillna = True) #add technical indicators
      df.drop(['open', 'high', 'low', 'adjclose', 'volume', 'ticker'], axis = 1, inplace = True) #drop irrelevant columns
      close_scaler = RobustScaler()
      close_scaler.fit(df[['close']])
      scaler = RobustScaler()
      df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns, index = df.index)

      n_features = df.shape[1]

      X, y = split_sequence(df.to_numpy(), n_per_in, n_per_out)

#==================== RNN MODEL ==================================================================================

      model = Sequential()
      act = "tanh"
      es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50) # PLAY WITH PATIENCE
      mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='auto' ,verbose=1, save_best_only=True) #PLAY WITH THIS

      #input layer will be an LSTM

      model.add(LSTM(90, #FIXME: 90 is hardcoded change this
                    activation=act,
                    return_sequences=True,
                    input_shape=(n_per_in, n_features)))

      #add hidden layers using previously created function

      layer_maker(model, n_layers = 1, 
                  n_nodes = 30,
                  activation = act)

      model.add(LSTM(60, activation = act)) #adds another LSTM layer

      model.add(Dense(n_per_out)) #adds a dense layer

      model.summary()

      model.compile(optimizer='adam', loss='mse', metrics=['accuracy']) #compiles the model

      res = model.fit(X, y, epochs=30, batch_size=128, validation_split=0.1, callbacks=[es, mc]) # PLAY WITH THESE PARAMETERS add callbacks=[es, mc] as parameter for early stopping

#==================== CREATES FRAME WITH PREDICTIONS AND ACTUAL ===================================================

      actual = pd.DataFrame(close_scaler.inverse_transform(df[["close"]]), 
                            index=df.index, 
                            columns=[df.columns[0]])

      predictions = validater(model, df, n_per_in, n_per_out, n_features, close_scaler)
  
#==================== CREATES GRAPH OF PREDICTED VS ACTUAL VALUES ==================================================

      fig = px.line(actual, x= df.index, y='close', color_discrete_sequence=["#0EBDEE"], width = 1000, height = 350, title = f"COMPARING PREDICTIONS TO REAL PRICES",
                    labels = "Actual Price")
      fig.add_scatter(x=predictions.index, y = predictions['close'], mode='lines', line=dict(color="#0000FF"))
      fig.update_xaxes(title_text='Time')
      fig.update_yaxes(title_text='Price')
      fig.update(layout_showlegend=False)
      st.plotly_chart(fig, use_container_width=True)
      st.write("Here, you can see a backtest of how the model performed when trying to predict previous time periods. In dark-blue, you can see the model's prediction. And in light-blue, you can visualize the real price of the stock at that time. The more they match up the better!")   

#==================== CREATE FORECASTING GRAPH ======================================================================

      yhat = model.predict(np.array(df.tail(n_per_in)).reshape(1, n_per_in, n_features))

      yhat = close_scaler.inverse_transform(yhat)[0]

      preds = pd.DataFrame(yhat, 
                          index=pd.date_range(start=df.index[-1]+timedelta(days=1), 
                                              periods=len(yhat), 
                                              freq="B"), 
                          columns=[df.columns[0]])

      pers = n_per_in

      actual = pd.DataFrame(close_scaler.inverse_transform(df[["close"]].tail(pers)), 
                            index=df.close.tail(pers).index, 
                            columns=[df.columns[0]]).append(preds.head(1))

      fig = px.line(actual, x= actual.index, y='close', color_discrete_sequence=["#0EBDEE"], width = 1000, height = 350, title = f"FORECASTING THE NEXT {len(yhat)} DAYS",
                    labels = "Actual Price")
      fig.add_scatter(x=preds.index, y = preds['close'], mode='lines', line=dict(color="#0000FF"))
      fig.update_xaxes(title_text='Time')
      fig.update_yaxes(title_text='Price')
      fig.update(layout_showlegend=False)
      st.plotly_chart(fig, use_container_width=True)

#==================== CALCULATES SLOPE OF PREDICTIONS AND WRITES DESCRIPTION OF PRICE BEHAVIOR ======================================================================   

      preds.insert(loc=0, column='index', value=np.arange(len(preds)))
      preds_array = preds.to_numpy()
      X = preds_array[:, 0]
      Y = preds_array[:, 1]
      z = np.polyfit(X, Y, 1)
      p = np.poly1d(z)
      
      actual = actual.append(preds)
      actual.drop(['index'], axis = 1, inplace = True)
      actual.insert(loc=0, column='index', value=np.arange(len(actual)))
      numpy_array = actual.to_numpy()
      X = numpy_array[numpy_array.shape[0]-n_per_out:, 0]
      Y = numpy_array[numpy_array.shape[0]-n_per_out:, 1]
      z1 = np.polyfit(X, Y, 1)
      p = np.poly1d(z1)
      if interval == 'daily':
        out_interval = 'DAYS'
      if interval == 'monthly':
        out_interval = 'MONTHS'
      if interval == 'weekly':
        out_interval = 'WEEKS'
      if z1[0] < 0:
        st.write(f"THIS STOCK IS PROBABLY GOING DOWN IN THE NEXT {n_per_out} " + out_interval)
        st.write("This means that based on several technical and fundamental indicators, our model noticed certain patterns that indicate a down trend in the price of this stock, for the future time period you chose.")
      if z1[0] > 0:
        st.write(f"THIS STOCK IS PROBABLY GOING UP IN THE NEXT {n_per_out} " + out_interval)
        st.write("This means that based on several technical and fundamental indicators, our model noticed certain patterns that indicate an up trend in the price of this stock, for the future time period you chose.")
    st.title("SENTIMENT ANALYSIS")


#=====================================================================================================================
#=================== SENTIMENT ANALYSIS ==============================================================================
#=====================================================================================================================

#==================== SENTIMENT ANALYSIS PROMPT ======================================================================

    ticker_sent = st.text_input("What do people think about this instrument?") 

#==================== EXECUTES ON BUTTON =============================================================================

    if st.button("LET'S GO!"): 

#==================== PARAMS FOR REQUEST ======================================================================

        params = {'q': ticker_sent,
                  'tweet_mode': 'extended',
                  'lang': 'en',
                  'count': '1000'
                  }
#==================== CREATES FRAME OF TWEETS ======================================================================

        tweets = requests.get(
            'https://api.twitter.com/1.1/search/tweets.json?',
            params=params,
            headers={
                'Authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAAM1QaQEAAAAAGNEmi%2Bw2EXjCf0wvfUU%2BCvDkqD4%3Dngsy6YrQVGCp0EuxLpFQPztczdksmje5puIb6Twmq6ywWxcX9P' 
        })


        def get_data_sent(tweet):
          data = {
              'id': tweet['id_str'],
              'created_at': tweet['created_at'],
              'text': tweet['full_text']
          }
          return data

        df = pd.DataFrame()
        sentiments = []
        probs = []
        pos = 0
        neg = 0
        
#==================== LOADS MODEL AND APPLIES IT TO TWEETS ======================================================================

        model = flair.models.TextClassifier.load('en-sentiment')

        for tweet in tweets.json()['statuses']:
          row = get_data_sent(tweet)
          df = df.append(row, ignore_index=True)

        df.tail()

        for sentence in df['text']:
          sentence = flair.data.Sentence(sentence)
          model.predict(sentence)
          probs.append(sentence.labels[0].score)
          sentiments.append(sentence.labels[0].value)
          if sentence.labels[0].value == 'NEGATIVE':
            neg = neg+1
          if sentence.labels[0].value == 'POSITIVE':
            pos = pos+1

#==================== CREATES DF OUT OF SENTIMENTS ======================================================================

        df['sentiment'] = sentiments #AO QUE PARECE POSSO SÓ ADICIONAR COLUNAS À DATA FRAME 
        df['probs'] = probs
        if neg < pos:
          st.write(f"Sentiment on this asset is overall positive! ({round(pos/(pos+neg)*100)}%)")
        if neg > pos:
          st.write(f"Sentiment on this asset is overall negative! ({round(neg/(pos+neg)*100)}%)")

        sent_data = [[pos, neg], [0,0]]
        sent = pd.DataFrame(sent_data, columns=['positive', 'negative'])
        feeling=['Sentiment']

#==================== CREATES GRAPH WITH SENTIMENTS ======================================================================

        fig = go.Figure(data=[
            go.Bar(name='Positive', x=feeling, y=[pos], marker = {'color' : '#AFD3F5'} ),
            go.Bar(name='Negative', x=feeling, y=[neg], marker = {'color' : '#006FC6'}) #D5003A
        ])

        fig.update_layout(template = "plotly_dark")
        # Change the bar modex
        fig.update_layout(barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("This tool gathers the most recent information from several social media sources like Twitter, Reddit, Instagram and Discord. Providing you an overall view of the world's opinion about this asset. Then we use a state-of-the-art model that classifies the content of those media as positive or negative, analyzing the outcome.")

show_predict_pge()
