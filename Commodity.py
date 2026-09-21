# -*- coding: utf-8 -*-
"""
=======================================================================================
Project Name: Simple Asset-Screening Analyzer (Custom Commodity Score!)
File Name:    Commodity.py
Author:       Andy Dinh
Date Created: September 20, 2026
Version:      1.1.0
Environment:  Python 3.13 or below (Stable Production Baseline)

Description:
    A vectorized asset-screening architecture that maps  
    historical commodity data. Calibrated around a thesis: 
    "The more factors, the better!" 
    Features future-proof version updates and 
    defensive error-catching gateways.

License & Copyright:
    Copyright (c) 2026 Andy Dinh. All Rights Reserved.
    PROPRIETARY AND CONFIDENTIAL. Strictly NO permission is granted for unauthorized 
    distribution, duplication, modification, or public cloning of this source logic.
=======================================================================================
"""


import subprocess
import sys
try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("\n YFINANCE NOT FOUND! INSTALLING NOW...\n")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "pandas"])
    import yfinance as yf
    import pandas as pd
#Auto Installation of yfinance and pandas if not found!


data = None
while(True):
    userinput = yf.Ticker(input("\nEnter the commodity's ticker symbols you want to track (Example: AAPL, TSLA, MSFT, etc.): \n"))
    periods = input("\nEnter the period you want to track (Options: 1y/1d/1mo, 2y, 3y/3mo, 5y/5d, 10y, ytd, max): \n")
    data = userinput.history(period=periods)
    if data.empty:
        print("\n (AS YOU CAN TELL FROM THE ERROR) - INVALID COMMODITY SYMBOL/PERIOD, TRY AGAIN.\n")
        continue
    else:
        print(f"\n Success! Here is your commodity({userinput.ticker}) data\n")
        break
#Asking The User!

period_lookup = {"1d": "a day", "5d": "5 days", "1mo": "a month", "3mo": "3 months", "1y": "a year", "2y": "2 years", "3y": "3 years", "5y": "5 years", "10y": "10 years", "ytd": "year to date", "max": "its entire lifetime"}
clean_period_name = period_lookup.get(periods.lower(), periods)
avg_price = data['Close'].mean()
daily_returns = data['Close'].pct_change().dropna()
avg_daily_return = daily_returns.mean()
std_dev = daily_returns.std()
ann_volatility = std_dev * (252 ** 0.5)
return_points = (avg_daily_return * 252) * 100
volatility_penalty = ann_volatility * 20
skew = daily_returns.skew()
if pd.isna(skew): 
    skew = 0
skew_penalty = (skew / 0.25)
avg_volume = userinput.info.get('averageVolume', 500000)
if avg_volume is None or avg_volume >= 500000:
    volume_penalty = 0
else:
    volume_penalty = ((1.2 ** ((500000 - avg_volume) / 25000)) - 1)
market_beta = userinput.info.get('beta', 1.0)
beta_penalty = 0
if market_beta is None: 
    market_beta = 1.0 
if market_beta > 1.2:
    steps_above = (market_beta - 1.2) / 0.2
    beta_penalty = -(1.5 ** steps_above) 
elif market_beta < 0.8:
    beta_penalty = (0.8 - market_beta) / 0.1 
score = return_points - volatility_penalty + skew_penalty - volume_penalty + beta_penalty
#Calculating Commodity Metrics!


print(f"\n-> The average closing price(how much it was!) over {clean_period_name} is: ${avg_price:.2f}\n")
print(f"-> The average daily return(how much you gained!) over {clean_period_name} is: {avg_daily_return * 100:.2f}%\n")
print(f"-> The annualized volatility(how volatile it is!) over {clean_period_name} is: {ann_volatility * 100:.2f}%\n")
print(f"-> The skewness(how asymmetrical it is!) of your commodity's daily returns is: {skew:.2f}\n")
print(f"-> The average daily trading volume(how much demand!) of your commodity is: {avg_volume}\n")
print(f"-> The beta(how sensitive it is to market changes!) of your commodity is: {market_beta:.2f}\n")
print("---------------------------------------------------------------------------------------\n")
print("-> We will now rate the commodity!\n")
print("-> Your commodity is judged based on various factors\n")
print("-> One point added for every 0.02% of daily returns\n")
print("-> One point subtracted for every 5% of annualized volatility\n")
print("-> One point added/subtracted for every 0.25/-0.25 of skewness\n")
print("-> 1.2ˣ points subtracted for every 25,000(x) of average daily volume below 500,000\n")
print("-> 1 point added for every 0.1 of beta below 0.8\n")
print("-> 1.5ˣ points subtracted for every 0.2(x) of beta above 1.2\n")
print("---------------------------------------------------------------------------------------\n")
print("-> Commodity Score Ratings:\n")
print("-> Less than -20? - Pack it up!\n")
print("-> -20 -> 0? - Playing with fire!\n")
print("-> 0 -> 10? - Okay, I see the vision!\n")
print("-> 10 -> 30? - Impressive, Warren Buffett!\n")
print("-> More than 30? - I'm gonna steal your commodity now!\n")
print("-> Your average daily return score is: {:.2f}\n".format(return_points))
print("-> Your volatility penalty is: -{:.2f}\n".format(volatility_penalty))
print("-> Your skewness penalty(or bonus) is: {:.2f}\n".format(skew_penalty))
print("-> Your volume penalty is: {:.2f}\n".format(volume_penalty * -1))
print("-> Your beta penalty(or bonus)is: {:.2f}\n".format(beta_penalty))
print(f"-> Your commodity score is: {score:.2f}!\n")
#Printing Commodity Metrics!