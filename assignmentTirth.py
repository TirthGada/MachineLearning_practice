#!/usr/bin/env python
# coding: utf-8

# In[87]:


import pandas as pd
from datetime import datetime

bank_nifty_data = pd.read_csv('banknifty_data.csv')
option_contract_data = pd.read_csv('option_contract_data.csv')

option_contract_data['<date>'] = option_contract_data['<date>'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y'))

active_trade = None
trades = []

def enter_long(row):
    return row['SMA']-row['<close>']< 0 and bank_nifty_data.iloc[index - 1]['SMA']- bank_nifty_data.iloc[index - 1]['<close>']>= 0

def exit_long(row):
    return row['Profit'] <= -20 or row['Profit'] >= 30

def enter_short(row):
    return row['SMA']- row['<close>'] >=0 and bank_nifty_data.iloc[index - 1]['SMA']-bank_nifty_data.iloc[index - 1]['<close>'] < 0

def exit_short(row):
    return row['Profit'] <= -20 or row['Profit'] >= 30

for index, row in option_contract_data.iterrows():
    option_date = row['<date>']
    option_time = row['<time>']
    print("Option Date:", option_date)
    print("Option Time:", option_time)

    matching_rows = bank_nifty_data[
        (bank_nifty_data['<date>'] == option_date.strftime('%d-%m-%Y')) & 
        (bank_nifty_data['<time>'] == option_time)
    ]
    
    print(matching_rows)
    if not matching_rows.empty:
        bank_nifty_row = matching_rows.iloc[0]
        
        if row['<time>'] >= '9:30:00' and row['<time>'] <= '15:00:00':
            if active_trade is None:
                if enter_long(bank_nifty_row):
                    active_trade = {'Type': 'Long', 'EntryPrice': row['<close>'], 'EntryTime': row['<time>']}
                elif enter_short(bank_nifty_row):
                    active_trade = {'Type': 'Short', 'EntryPrice': row['<close>'], 'EntryTime': row['<time>']}
            elif active_trade['Type'] == 'Long':
                active_trade['Profit'] = row['<close>'] - active_trade['EntryPrice']
                if exit_long(active_trade) or row['<time>'] == '15:00:00':
                    active_trade['ExitTime'] = row['<time>']
                    trades.append(active_trade)
                    active_trade = None
            elif active_trade['Type'] == 'Short':
                active_trade['Profit'] = active_trade['EntryPrice'] - row['<close>']
                if exit_short(active_trade) or row['<time>'] == '15:00':
                    active_trade['ExitTime'] = row['<time>']
                    trades.append(active_trade)
                    active_trade = None
    
       

total_trades = len(trades)
profitable_trades = len([trade for trade in trades if trade['Profit'] >= 0])
success_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
total_profit = sum(trade['Profit'] for trade in trades)

trades_df = pd.DataFrame(trades)

trades_df.to_csv('trades.csv', index=False)

print("Total Trades:", total_trades)
print("Profitable Trades:", profitable_trades)
print("Success Rate:", success_rate)
print("Total Profit:", total_profit)


# In[20]:


bank_nifty_data


# In[21]:


option_contract_data

