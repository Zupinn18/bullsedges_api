from pya3 import *
import pandas as pd
import time
import csv
import logging

# Initialize Aliceblue
user_id = 'AB093838'
api_key = 'cy5uYssgegMaUOoyWy0VGLBA6FsmbxYd0jNkajvBVJuEV9McAM3o0o2yG6Z4fEFYUGtTggJYGu5lgK89HumH3nBLbxsLjgplbodFHDLYeXX0jGQ5CUuGtDvYKSEzWSMk'

# Initialize AliceBlue connection
alice = Aliceblue(user_id=user_id, api_key=api_key)
print(alice.get_session_id())

# Path to the CSV files
label_csv_file_path = 'label_39321PE.csv'
# adx_csv_file_path = 'adx_PE_data.csv'

# Initialize the state variable
state = 'waiting_for_yes'
yes_order_executed = False  # Add a flag to track if a "YES" order has been executed

# Flags to track buy and sell orders
buy_order_placed = False
sell_order_placed = False

# Function to place a Buy order
def place_buy_order():
    global buy_order_placed, sell_order_placed  # Use global variables to track order statuses
    print("Placing a Buy order...")
    print(alice.place_order(
        transaction_type=TransactionType.Buy,
        instrument=alice.get_instrument_by_token('NFO', 39321),
        quantity=15,
        order_type=OrderType.Market,
        product_type=ProductType.Intraday,
        price=0.0,
        trigger_price=None,
        stop_loss=None,
        square_off=None,
        trailing_sl=None,
        is_amo=False,
        order_tag='order1'
    ))
    buy_order_placed = True  # Set the buy order flag to True
    sell_order_placed = False  # Reset the sell order flag

# Function to place a Sell order
def place_sell_order():
    global sell_order_placed  # Use a global variable to track the sell order status
    if buy_order_placed:
        print("Placing a Sell order...")
        print(alice.place_order(
            transaction_type=TransactionType.Sell,
            instrument=alice.get_instrument_by_token('NFO', 39321),
            quantity=15,
            order_type=OrderType.Market,
            product_type=ProductType.Intraday,
            price=0.0,
            trigger_price=None,
            stop_loss=None,
            square_off=None,
            trailing_sl=None,
            is_amo=False,
            order_tag='order1'
        ))
        sell_order_placed = True  # Set the sell order flag to True
    else:
        print("Cannot place Sell order. Buy order has not been placed.")

timer = 0

log_file = 'trading_logs_PE.csv'
csv_log_file = open(log_file, mode='w', newline='')
csv_writer = csv.writer(csv_log_file)
csv_writer.writerow(['Timestamp', 'Level', 'Message'])  # Write the header row

def write_log_entry(timestamp, level, message):
    csv_writer.writerow([timestamp, level, message])
    csv_log_file.flush()

while True:
    try:
        # Read the CSV file for label_26009.csv
        df_label = pd.read_csv(label_csv_file_path)
        # Read the CSV file for adx_filtered_data.csv
        # df_adx = pd.read_csv(adx_csv_file_path)

        # Check if the DataFrames are empty
        # if df_label.empty or df_adx.empty:
        if df_label.empty:
            print("CSV file is empty. Waiting for data...")
        else:
            # Check the last row of the label CSV for 'yes' value
            last_row_label = df_label.iloc[-1]
            value_label = last_row_label['Label']  # Replace with the actual column name
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Check the last row of the ADX CSV for conditions
            # last_row_adx = df_adx.iloc[-1]
            # adx_value = last_row_adx['adx']  # Replace with the actual column name
            # di_color = last_row_adx['+di_color']  # Replace with the actual column name

            # if state == 'waiting_for_yes' and value_label == 'YES' and adx_value > 20 and di_color == 'up':
            if state == 'waiting_for_yes' and value_label == 'YES':
                write_log_entry(timestamp, 'INFO', "State: Waiting for yes and conditions met")
                # Place a Buy order and reset the timer
                place_buy_order()
                write_log_entry(timestamp, 'INFO', "Buy order placed...")
                state = 'waiting_for_action'
                write_log_entry(timestamp, 'INFO', "Waiting for action...")
                yes_order_executed = True  # Set the flag to True when a "YES" order is executed
                write_log_entry(timestamp, 'INFO', "Yes order executed...")
                timer = 0  # Reset the timer
            elif state == 'waiting_for_action':
                write_log_entry(timestamp, 'INFO', "State: Waiting for action")
                if value_label == 'seven':
                    # Place a Sell order only if a Buy order has been placed and reset the timer
                    place_sell_order()
                    write_log_entry(timestamp, 'INFO', "Sell order executed, market hit 7+ points")
                    state = 'waiting_for_yes'  # Reset the state
                    timer = 0  # Reset the timer
                elif value_label == 'RED':
                    place_sell_order()
                    time.sleep(300)  # Wait for 5 minutes (300 seconds) before checking the new label
                    state = 'waiting_for_yes'  # Reset the state
                    write_log_entry(timestamp, 'INFO', "State: waiting for yes after sell order executed")
                # elif value == 'low':
                #     place_sell_order()
                #     state = 'waiting_for_yes'  # Reset the state
            else:
                print("Invalid value in CSV:", value_label)
    except Exception as e:
        print("An error occurred:", str(e))

    # Sleep for a while before checking again
    time.sleep(1)
