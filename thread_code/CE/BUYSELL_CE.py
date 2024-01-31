from pya3 import *
import pandas as pd
import time
import csv
import logging
import threading

# Initialize Aliceblue
user_id = 'AB093838'
api_key = 'cy5uYssgegMaUOoyWy0VGLBA6FsmbxYd0jNkajvBVJuEV9McAM3o0o2yG6Z4fEFYUGtTggJYGu5lgK89HumH3nBLbxsLjgplbodFHDLYeXX0jGQ5CUuGtDvYKSEzWSMk'

# Initialize AliceBlue connection
alice = Aliceblue(user_id=user_id, api_key=api_key)
print(alice.get_session_id())

# Path to the CSV file
csv_file_path = 'label_40707CE_02.csv'

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
        instrument=alice.get_instrument_by_token('NFO', 40707),
        quantity=0,
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
            instrument=alice.get_instrument_by_token('NFO', 40707),
            quantity=0,
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

log_file = 'trading_logs_CE.csv'
csv_log_file = open(log_file, mode='w', newline='')
csv_writer = csv.writer(csv_log_file)
csv_writer.writerow(['Timestamp', 'Level', 'Message'])  # Write the header row

def write_log_entry(timestamp, level, message):
    csv_writer.writerow([timestamp, level, message])
    csv_log_file.flush()

def check_for_seven_or_no():
    global state
    while True:
        try:
            if state == 'waiting_for_action':
                if value == 'seven':
                    place_sell_order()
                    state = 'waiting_for_yes'
                    timer = 0
            time.sleep(1)
        except Exception as e:
            print("An error occurred:", str(e))

# Function to sleep for 250 seconds
def sleep_for_250_seconds():
    global timer
    while timer < 250:
        time.sleep(1)
        timer += 1
    if not sell_order_placed:
        place_sell_order()

# Create and start threads
check_seven_or_no_thread = threading.Thread(target=check_for_seven_or_no)
check_seven_or_no_thread.start()

while True:
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)

        # Check if the DataFrame is empty
        if df.empty:
            if state == 'waiting_for_action':
                if yes_order_executed:
                    # If the state is 'waiting_for_action' (after a "YES" order)
                    # execute a sell order and reset the state
                    time.sleep(250)  # Wait for 30 seconds
                    place_sell_order()
                    yes_order_executed = False  # Reset the flag
                    state = "waiting_for_yes"
            else:
                print("CSV file is empty. Waiting for data...")
        else:
            # Check the last row of the CSV for 'yes', 'no', or 'ten' value
            last_row = df.iloc[-1]
            value = last_row['Label']  # Replace with the actual column name
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            if state == 'waiting_for_yes' and value == 'YES':
                write_log_entry(timestamp, 'INFO', "State: Waiting for yes")
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
                if value == 'seven':
                    # Place a Sell order only if a Buy order has been placed and reset the timer
                    write_log_entry(timestamp, 'INFO', "Sell order executed, market hit 7+ points")
                    place_sell_order()
                    state = 'waiting_for_yes'  # Reset the state
                    timer = 0  # Reset the timer
                elif value == 'seven_down':
                    # Place a Sell order only if a Buy order has been placed and reset the timer
                    write_log_entry(timestamp, 'INFO', "Sell order executed, market goes 7 points down")
                    place_sell_order()
                    state = 'waiting_for_yes'  # Reset the state
                    timer = 0  # Reset the timer
                elif value == 'NO':
                    time.sleep(290)
                    last_row = df.iloc[-1]
                    value = last_row['Label']
                    if value == 'YES':
                        state = "waiting_for_action"
                        write_log_entry(timestamp, 'INFO', "No is removed waiting for action")
                    elif value == 'NO':
                        write_log_entry(timestamp, 'INFO', "Sell order executed(NO)...")
                        place_sell_order()
                        state = 'waiting_for_yes'  # Reset the state
                    state = 'waiting_for_yes'  # Reset the state
                    write_log_entry(timestamp, 'INFO', "State: waiting for yes after sell order executed")
                # elif value == 'low':
                #     place_sell_order()
                #     state = 'waiting_for_yes'  # Reset the state
            else:
                print("Invalid value in CSV:", value)
    except Exception as e:
        print("An error occurred:", str(e))

    # Sleep for a while before checking again
    time.sleep(1)  