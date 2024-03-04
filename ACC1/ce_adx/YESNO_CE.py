import json
from datetime import datetime, timedelta
from pytz import timezone
from time import sleep
import pandas as pd
import dash
import os
import csv
# from telegram import Bot  # Import the Bot class from the telegram module
# from telegram.error import TelegramError
from dash import dcc
from dash import html
from dash.dependencies import Output, Input, State
import streamlit as st
import plotly.graph_objects as go
from pya3 import *
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import dash_table
import pymongo
from pymongo import MongoClient
import ta
from ta.utils import dropna



# Replace these with your actual MongoDB connection details
MONGO_CONNECTION_STRING = "mongodb://localhost:27017/"
DB_NAME = "banknifty"
COLLECTION_NAME = "39328CE"

client = MongoClient(MONGO_CONNECTION_STRING)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Define your AliceBlue user ID and API key
user_id = 'AB093838'
api_key = 'cy5uYssgegMaUOoyWy0VGLBA6FsmbxYd0jNkajvBVJuEV9McAM3o0o2yG6Z4fEFYUGtTggJYGu5lgK89HumH3nBLbxsLjgplbodFHDLYeXX0jGQ5CUuGtDvYKSEzWSMk'

# Initialize AliceBlue connection
alice = Aliceblue(user_id=user_id, api_key=api_key)

# Print AliceBlue session ID
print(alice.get_session_id())

# Initialize variables for WebSocket communication
lp = 0
socket_opened = False
subscribe_flag = False
subscribe_list = []
unsubscribe_list = []
data_list = []  # List to store the received data
df = pd.DataFrame(columns=["timestamp", "lp"])  # Initialize an empty DataFrame for storing the data
# File paths for saving data and graph
data_file_path = "39328CE.csv"

graph_file_path = "39328CE.html"

# Check if the data file exists
if os.path.exists(data_file_path):
    # Load existing data from the CSV file
    df = pd.read_csv(data_file_path, index_col="timestamp", parse_dates=True)
else:
    df = pd.DataFrame(columns=["timestamp", "lp"])  # Initialize an empty DataFrame for storing the data


all_trend_lines = []
trend_line_visibility = []


# Callback functions for WebSocket connection
def socket_open():
    print("Connected")
    global socket_opened
    socket_opened = True
    if subscribe_flag:
        alice.subscribe(subscribe_list)


def socket_close():
    global socket_opened, lp
    socket_opened = False
    lp = 0
    print("Closed")


def socket_error(message):
    global lp
    lp = 0
    print("Error:", message)

consecutive_green_candles = 0
previous_candle_green = False
label_data = []

# Callback function for receiving data from WebSocket
def feed_data(message):
    global lp, subscribe_flag, data_list, consecutive_green_candles
    feed_message = json.loads(message)
    if feed_message["t"] == "ck":
        print("Connection Acknowledgement status: %s (Websocket Connected)" % feed_message["s"])
        subscribe_flag = True
        print("subscribe_flag:", subscribe_flag)
        print("-------------------------------------------------------------------------------")
        pass
    elif feed_message["t"] == "tk":
        print("Token Acknowledgement status: %s" % feed_message)
        print("-------------------------------------------------------------------------------")
        pass
    else:
        print("Feed:", feed_message)
        if 'lp' in feed_message:
            timestamp = datetime.now(timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
            feed_message['timestamp'] = timestamp
            data_list.append(feed_message)  # Append the received data to the list
            # Insert the data into MongoDB
            collection.insert_one(feed_message)

            # Update marking information only for Heikin Ashi candles
            if len(df) >= 2 and df['mark'].iloc[-1] == '' and feed_message['t'] == 'c':
                if (df['close'].iloc[-2] > df['open'].iloc[-2] and df['close'].iloc[-1] > df['open'].iloc[-1]):
                    if previous_candle_green:  # Check if the previous candle was green
                        consecutive_green_candles += 1
                        if consecutive_green_candles == 2:  # Mark "YES" only on the second consecutive green candle
                            df.at[df.index[-1], 'mark'] = 'YES'
                    else:
                        consecutive_green_candles = 1
                    previous_candle_green = True
                elif (df['close'].iloc[-2] > df['open'].iloc[-2] and df['close'].iloc[-1] < df['open'].iloc[-1]):
                    consecutive_green_candles = 0
                    previous_candle_green = False
                    df.at[df.index[-1], 'mark'] = 'NO'

        else:
            print("'lp' key not found in feed message.")


# Connect to AliceBlue

# Socket Connection Request
alice.start_websocket(socket_open_callback=socket_open, socket_close_callback=socket_close,
                      socket_error_callback=socket_error, subscription_callback=feed_data, run_in_background=True,
                      market_depth=False)

while not socket_opened:
    pass

# Subscribe to Tata Motors
subscribe_list = [alice.get_instrument_by_token('NFO', 39328)]
alice.subscribe(subscribe_list)
print(datetime.now())
sleep(10)
print(datetime.now())

def calculate_heikin_ashi(data):
    ha_open = 0.5 * (data['open'].shift() + data['close'].shift())
    ha_close = 0.25 * (data['open'] + data['high'] + data['low'] + data['close'])
    ha_high = data[['high', 'open', 'close']].max(axis=1)
    ha_low = data[['low', 'open', 'close']].min(axis=1)

    ha_data = pd.DataFrame({'open': ha_open, 'high': ha_high, 'low': ha_low, 'close': ha_close})
    
    for i in range(len(ha_data)):
        if i == 0:
            ha_data.iat[0, 0] = round(((data['open'].iloc[0] + data['close'].iloc[0]) / 2), 2)
        else:
            ha_data.iat[i, 0] = round(((ha_data.iat[i-1, 0] + ha_data.iat[i-1, 3]) / 2), 2)

    ha_data['high'] = ha_data[['high', 'open', 'close']].max(axis=1)
    ha_data['low'] = ha_data[['low', 'open', 'close']].min(axis=1)
    ha_data['close'] = round(0.25 * (data['open'] + data['close'] + data['high'] + data['low']), 2)

    ha_data['mark'] = ''
    label_data = []

    # Initialize state
    consecutive_green_candles = 0
    prev_yes_open = None
    prev_green_low = None  # Track the low of the previous green candle
    prev_green_high = None 
    no_confirmed = True  # Flag to track if "NO" has been confirmed
    # yes_updated = False
    label_data = []  # Create an empty list to store label data

    for i in range(1, len(ha_data)):
        if (ha_data['close'].iloc[i - 1] > ha_data['open'].iloc[i - 1] and
                ha_data['close'].iloc[i] > ha_data['open'].iloc[i]):
            consecutive_green_candles += 1
            prev_yes_open = data['open'].iloc[i]  # Update previous "YES" open value
            prev_green_high = ha_data['high'].iloc[i]  # Update the high of the previous green candle

            if consecutive_green_candles == 1:  # Change made here, check for second green candle
                ha_data.at[ha_data.index[i], 'mark'] = 'YES'
                label_data.append(('YES', ha_data.index[i], data['open'].iloc[i], None))
                # Check if the current closing price is 7 points higher than the previous "YES" high
                if prev_green_high is not None and data['high'].iloc[i] > ha_data['high'].iloc[i - 1] + 7:
                    label_data.append(('seven', ha_data.index[i], data['high'].iloc[i], None))
                consecutive_green_candles = 0  # Reset consecutive green candles count
        elif (ha_data['close'].iloc[i - 1] > ha_data['open'].iloc[i - 1] and
            ha_data['close'].iloc[i] < ha_data['open'].iloc[i]):
            # Check if the previous candle was green
            if consecutive_green_candles > 0:
                ha_data.at[ha_data.index[i], 'mark'] = 'NO'
                if prev_yes_open is not None:
                    if prev_green_high is not None and data['high'].iloc[i] > ha_data['high'].iloc[i - 1] + 7:
                        label_data.append(('seven', ha_data.index[i], data['high'].iloc[i], None))
                    if no_confirmed:  # Calculate difference only if "NO" is confirmed
                        confirmed_no_closing = ha_data['close'].iloc[i]  # Store confirmed "NO" closing value
                        diff = prev_yes_open - confirmed_no_closing  # Corrected difference calculation
                        label_data.append(('NO', ha_data.index[i], confirmed_no_closing, diff))
                        
                        if ha_data['close'].iloc[i] < ha_data['open'].iloc[i]:
                            prev_green_low = ha_data['low'].iloc[i]  # Update the low of the previous green candle

                        if i + 1 < len(ha_data) and ha_data['close'].iloc[i + 1] < ha_data['open'].iloc[i + 1]:
                            label_data.append(('RED', ha_data.index[i + 1], ha_data['close'].iloc[i + 1], None))
                    else:
                        ha_data.at[ha_data.index[i], 'mark'] = ''
                        print("Warning: NO not confirmed yet, skipping difference calculation")
                else:
                    ha_data.at[ha_data.index[i], 'mark'] = ''
                    print("Warning: prev_yes_open is None, skipping difference calculation")

                no_confirmed = True  # "NO" is confirmed
                consecutive_green_candles = 0  # Reset consecutive_green_candles

            

    # Calculate the difference and add it to the DataFrame
    ha_data['Difference'] = ha_data['open'] - ha_data['close']

    label_csv_filename = 'label_39328CE.csv'
    try:
        with open(label_csv_filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Label', 'Timestamp', 'Value', 'Difference'])
            csv_writer.writerows(label_data)
        print(f'Labels saved to {label_csv_filename}')
    except Exception as e:
        print(f'Error saving labels: {e}')

    return ha_data


def calculate_he_adx(data, period=14):
    ha_data = calculate_heikin_ashi(data)

    # Calculate ADX, +DI, and -DI using ta library
    adx_indicator = ta.trend.ADXIndicator(ha_data['high'], ha_data['low'], ha_data['close'], window=period, fillna=True)
    ha_data['adx'] = adx_indicator.adx()
    ha_data['plus_di'] = adx_indicator.adx_pos()
    ha_data['minus_di'] = adx_indicator.adx_neg()

    # Find the index where ADX first crosses 20
    idx = (ha_data['adx'] > 20).idxmax()

    # Extract relevant data
    adx_cross_data = ha_data.loc[idx:, ['adx', 'plus_di', 'minus_di']]

    # Determine color for +DI and -DI
    adx_cross_data['+di_color'] = 'up'
    adx_cross_data.loc[adx_cross_data['plus_di'] < adx_cross_data['minus_di'], '+di_color'] = 'down'
    adx_cross_data['-di_color'] = 'up'
    adx_cross_data.loc[adx_cross_data['plus_di'] > adx_cross_data['minus_di'], '-di_color'] = 'down'

    # Save data to CSV file
    adx_cross_data.to_csv('adx_CE_data.csv', index_label='timestamp')

    return adx_cross_data[['adx', 'plus_di', 'minus_di']]


# Example usage:
# filtered_data = calculate_he_adx(data)



# def calculate_supertrend(data, atr_period=12, factor=2.0, multiplier=2.0):
#     data = data.copy()  # Create a copy of the data DataFrame

#     close = data['close']
#     high = data['high']
#     low = data['low']

#     tr = pd.DataFrame()
#     tr['h-l'] = high - low
#     tr['h-pc'] = abs(high - close.shift())
#     tr['l-pc'] = abs(low - close.shift())
#     tr['tr'] = tr.max(axis=1)

#     atr = tr['tr'].rolling(atr_period).mean()

#     median_price = (high + low) / 2
#     data['upper_band'] = median_price + (multiplier * atr)
#     data['lower_band'] = median_price - (multiplier * atr)

#     supertrend = pd.Series(index=data.index)
#     direction = pd.Series(index=data.index)

#     supertrend.iloc[0] = data['upper_band'].iloc[0]
#     direction.iloc[0] = 1

#     for i in range(1, len(data)):
#         if close.iloc[i] > supertrend.iloc[i - 1]:
#             supertrend.iloc[i] = max(data['lower_band'].iloc[i], supertrend.iloc[i - 1])
#             direction.iloc[i] = 1
#         else:
#             supertrend.iloc[i] = min(data['upper_band'].iloc[i], supertrend.iloc[i - 1])
#             direction.iloc[i] = -1

#         # Start uptrend calculation anew whenever a new uptrend begins
#         if direction.iloc[i] == 1 and direction.iloc[i - 1] != 1:
#             supertrend.iloc[i] = data['lower_band'].iloc[i]

#         # Start downtrend calculation anew whenever a new downtrend begins
#         if direction.iloc[i] == -1 and direction.iloc[i - 1] != -1:
#             supertrend.iloc[i] = data['upper_band'].iloc[i]

#     data['supertrend'] = supertrend  # Add the 'supertrend' column to the data DataFrame
#     data['direction'] = direction  # Add the 'direction' column to the data DataFrame

#     return data[['open', 'high', 'low', 'close', 'supertrend', 'direction', 'lower_band', 'upper_band']]

# def calculate_trend_lines(data):
#     current_trend = None
#     trend_start = None
#     trend_lines = []

#     for i in range(len(data)):
#         current_signal = data.iloc[i]

#         if current_trend is None:
#             current_trend = current_signal['direction']
#             trend_start = current_signal.name

#         if current_trend != current_signal['direction']:
#             if trend_start is not None:
#                 trend_data = data.loc[trend_start:data.index[i - 1]]
#                 if len(trend_data) > 1:
#                     trend_lines.append((current_trend, trend_data))

#             current_trend = current_signal['direction']
#             trend_start = current_signal.name

#     # Handle the last trend if it's still ongoing
#     if trend_start is not None and trend_start != data.index[-1]:
#         trend_data = data.loc[trend_start:]
#         if len(trend_data) > 1:
#             trend_lines.append((current_trend, trend_data))

#     return trend_lines


# all_trend_lines = []



# # Function to update the graph

# def calculate_current_trend_lines(data):
#     current_trend = None
#     in_trend = False
#     trend_start = None
#     trend_lines = []
#     buy_signals = pd.DataFrame(columns=['supertrend', 'direction'])
#     sell_signals = pd.DataFrame(columns=['supertrend', 'direction'])
#     # band_data = pd.DataFrame(columns=['Timestamp', 'band_type', 'band_value'])

#     for i in range(len(data)):
#         current_signal = data.iloc[i]

#         if current_trend is None:
#             current_trend = current_signal['direction']
#             in_trend = True
#             trend_start = current_signal.name

#         if current_trend != current_signal['direction']:
#             if current_signal['direction'] == 1:
#                 sell_signals = pd.concat([sell_signals, current_signal])
#                 # band_data = pd.concat([band_data, pd.DataFrame({'Timestamp': [current_signal.name], 'band_type': ['lower_band'], 'band_value': [current_signal['lower_band']]}), pd.DataFrame({'Timestamp': [current_signal.name], 'band_type': ['upper_band'], 'band_value': [np.nan]})])
#             else:
#                 buy_signals = pd.concat([buy_signals, current_signal])
#                 # band_data = pd.concat([band_data, pd.DataFrame({'Timestamp': [current_signal.name], 'band_type': ['upper_band'], 'band_value': [current_signal['upper_band']]}), pd.DataFrame({'Timestamp': [current_signal.name], 'band_type': ['lower_band'], 'band_value': [np.nan]})])

#             if in_trend:
#                 trend_data = data.loc[trend_start:data.index[i - 1]]

#                 if len(trend_data) > 1:
#                     trend_lines.append((current_trend, trend_data))

#             else:
#                 if current_signal['direction'] == 1 and current_trend == -1:
#                     updated_trend_data = data.loc[trend_start:data.index[i]]
#                     updated_supertrend_data = calculate_supertrend(updated_trend_data, factor=2.0)
#                     current_signal['supertrend'] = updated_supertrend_data['supertrend'].iloc[-1]
#                     current_signal['direction'] = updated_supertrend_data['direction'].iloc[-1]
#                 elif current_signal['direction'] == -1 and current_trend == 1:
#                     updated_trend_data = data.loc[trend_start:data.index[i]]
#                     updated_supertrend_data = calculate_supertrend(updated_trend_data, factor=2.0)
#                     current_signal['supertrend'] = updated_supertrend_data['supertrend'].iloc[-1]
#                     current_signal['direction'] = updated_supertrend_data['direction'].iloc[-1]

#             current_trend = current_signal['direction']
#             in_trend = False

#         if not in_trend:
#             if current_trend == 1:
#                 if not np.isnan(current_signal['upper_band']):
#                     trend_start = current_signal.name
#                     in_trend = True
#             else:
#                 if not np.isnan(current_signal['lower_band']):
#                     trend_start = current_signal.name
#                     in_trend = True

#     if in_trend:
#         trend_data = data.loc[trend_start:]

#         if len(trend_data) > 1:
#             first_high = trend_data['high'].iloc[0]
#             last_close = trend_data['close'].iloc[-1]

#     # Handle the continuation of uptrend without a change in direction
#     if len(trend_lines) > 0 and data.index[-1] not in trend_lines[-1][1].index and trend_lines[-1][0] == 1:
#         last_trend_type, last_trend_data = trend_lines[-1]
#         continuation_data = data.loc[data.index > last_trend_data.index[-1]]
#         if len(continuation_data) > 1:
#             updated_trend_data = pd.concat([last_trend_data.iloc[:-1], continuation_data])
#             updated_supertrend_data = calculate_supertrend(updated_trend_data, factor=2.0)
#             continuation_data['supertrend'] = updated_supertrend_data['supertrend'].values[-len(continuation_data):]
#             continuation_data['direction'] = updated_supertrend_data['direction'].values[-len(continuation_data):]
#             trend_lines[-1] = (last_trend_type, updated_trend_data)

#     # Save band_data to a single CSV file
#     # band_data.to_csv('band_data_CE.csv', index=False)

#     return trend_lines, buy_signals, sell_signals

def calculate_adx(df):
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14, fillna=True).adx()
    df['di_plus'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14, fillna=True).adx_pos()
    df['di_minus'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14, fillna=True).adx_neg()

    return df
app = dash.Dash(__name__)
server = app.server

# Add a new dropdown for selecting candle type
candle_type_dropdown = dcc.Dropdown(
    id='candle-type-dropdown',
    options=[
        {'label': 'Normal Candlestick', 'value': 'normal'},
        {'label': 'Heikin Ashi', 'value': 'heikin_ashi'},
    ],
    value='normal',
    style={'width': '50%'},
)

# Add a new graph for Heikin Ashi candlestick
heikin_ashi_graph = dcc.Graph(id='heikin-ashi-graph')
adx_graph = dcc.Graph(id='adx-graph')
heikin_ashi_adx_graph = dcc.Graph(id='heikin-ashi-adx-graph')

# Update layout to include the new graph component
app.layout = html.Div([
    candle_type_dropdown,
    dcc.Graph(id='candlestick-graph'),
    heikin_ashi_graph,
    adx_graph,
    heikin_ashi_adx_graph,  # Add the new graph component
    dcc.Dropdown(
        id='timeframe-dropdown',
        options=[
            {'label': '1 min', 'value': '1T'},
            {'label': '3 min', 'value': '3T'},
            {'label': '5 min', 'value': '5T'},
            {'label': '10 min', 'value': '10T'},
            {'label': '15 min', 'value': '15T'},
            {'label': '30 min', 'value': '30T'},
            {'label': '1 hr', 'value': '1H'},
        ],
        value='5T',  # Default to 5 min
        style={'width': '50%'},
    ),
    dcc.Interval(id='update-interval', interval=5*1000, n_intervals=0)
])

# Callback to update the candlestick graphs based on selected time frame and candle type
# Callback to update the candlestick graphs based on selected time frame and candle type
@app.callback(
    [Output('candlestick-graph', 'figure'),
     Output('heikin-ashi-graph', 'figure'),
     Output('adx-graph', 'figure'),
     Output('heikin-ashi-adx-graph', 'figure')],  # Add output for the new graph
    [Input('update-interval', 'n_intervals'),
     Input('candlestick-graph', 'relayoutData')],
    [State('timeframe-dropdown', 'value'),
     State('candle-type-dropdown', 'value')]
)
def update_graph_callback(n, relayoutData, selected_timeframe, selected_candle_type):
    data = collection.find({}, {'_id': 0}).sort('timestamp')
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    df.set_index('timestamp', inplace=True)

    # Drop non-numeric values in the 'lp' column
    df['lp'] = pd.to_numeric(df['lp'], errors='coerce')
    df = df.dropna(subset=['lp'])

    # Handle xaxis range if available in relayoutData
    # Handle xaxis range if available in relayoutData
    if 'xaxis.range' in relayoutData:
        xaxis_range = relayoutData['xaxis.range']
    else:
        # Adjust xaxis_range to include the last 4320 minutes
        end_time = df.index[-1]
        start_time = end_time - pd.Timedelta(hours=72)
        xaxis_range = [start_time, end_time]

    # Filter data based on xaxis range
    filtered_data = df[(df.index >= xaxis_range[0]) & (df.index <= xaxis_range[1])]


    # Resample data for candlestick graph based on the selected timeframe
    resampled_data = filtered_data["lp"].resample(selected_timeframe).ohlc()
    resampled_data = resampled_data.dropna()

    # Calculate ADX for the resampled data
    adx_data = calculate_adx(resampled_data)

    # Create normal candlestick figure
    normal_candlestick_fig = go.Figure(data=[go.Candlestick(x=resampled_data.index,
                                                             open=resampled_data['open'],
                                                             high=resampled_data['high'],
                                                             low=resampled_data['low'],
                                                             close=resampled_data['close'])])
    normal_candlestick_fig.update_xaxes(type='category', tickformat='%H:%M')
    normal_candlestick_fig.update_layout(title=f'Real-Time Candlestick Chart ({selected_timeframe})',
                                  xaxis_title='Time',
                                  yaxis_title='Price',
                                  xaxis_rangeslider_visible=False,
                                  template='plotly')

    # Create Heikin Ashi candlestick figure
    if selected_candle_type == 'heikin_ashi':
        heikin_ashi_data = calculate_heikin_ashi(resampled_data)
        heikin_ashi_fig = go.Figure(data=[go.Candlestick(x=heikin_ashi_data.index,
                                                         open=heikin_ashi_data['open'],
                                                         high=heikin_ashi_data['high'],
                                                         low=heikin_ashi_data['low'],
                                                         close=heikin_ashi_data['close'])])
        heikin_ashi_fig.update_xaxes(type='category', tickformat='%H:%M')
        heikin_ashi_fig.update_layout(title=f'Real-Time Heikin Ashi Candlestick Chart ({selected_timeframe})',
                                  xaxis_title='Time',
                                  yaxis_title='Price',
                                  xaxis_rangeslider_visible=False,
                                  template='plotly')
    else:
        heikin_ashi_fig = go.Figure()

    # Create ADX figure
    adx_fig = go.Figure(data=[go.Scatter(x=adx_data.index, y=adx_data['adx'], mode='lines', name='ADX')])
    # Add +DI trace
    adx_fig.add_trace(go.Scatter(x=adx_data.index, y=adx_data['di_plus'], mode='lines', name='+DI', line=dict(color='green')))

    # Add -DI trace
    adx_fig.add_trace(go.Scatter(x=adx_data.index, y=adx_data['di_minus'], mode='lines', name='-DI', line=dict(color='red')))
    adx_fig.update_xaxes(type='category', tickformat='%H:%M')
    adx_fig.update_layout(title=f'Average Directional Index (ADX) ({selected_timeframe})',
                                  xaxis_title='Time',
                                  yaxis_title='ADX Value',
                                  template='plotly')
    
    heikin_ashi_adx_data = calculate_he_adx(resampled_data)

    # Create Heikin Ashi ADX figure
    heikin_ashi_adx_fig = go.Figure(data=[
    go.Scatter(x=heikin_ashi_adx_data.index, y=heikin_ashi_adx_data['adx'], mode='lines', name='ADX', line=dict(color='blue')),
    go.Scatter(x=heikin_ashi_adx_data.index, y=heikin_ashi_adx_data['plus_di'], mode='lines', name='+DI', line=dict(color='green')),
    go.Scatter(x=heikin_ashi_adx_data.index, y=heikin_ashi_adx_data['minus_di'], mode='lines', name='-DI', line=dict(color='red'))
])

    heikin_ashi_adx_fig.update_xaxes(type='category', tickformat='%H:%M')
    heikin_ashi_adx_fig.update_layout(title=f'Heikin Ashi Average Directional Index (ADX) ({selected_timeframe})',
                                       xaxis_title='Time',
                                       yaxis_title='ADX Value',
                                       template='plotly')

    return normal_candlestick_fig, heikin_ashi_fig, adx_fig, heikin_ashi_adx_fig


if __name__ == '__main__':
    app.run_server(debug=True)
