# -*- coding: utf-8 -*-
"""
=======================================================================================
Project Name: Simple Asset-Screening Analyzer (Custom Commodity Score!)
File Name:    Commodity.py
Author:       Andy Dinh
Date Created: September 20, 2026
Version:      1.1.1
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
import math
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError:
    import subprocess
    import sys
    print("\n  INSTALLING NECESSARY STUFF NOW...\n")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "pandas", "nltk", "numpy"])
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import yfinance as yf
    import pandas as pd
    import numpy as np
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
#Auto Installation if not found!

data = None
running = True
while running:
    userinput = yf.Ticker(input("\nEnter the commodity's ticker symbols you want to track (Example: AAPL, TSLA, MSFT, etc.): \n"))
    periods = input("\nEnter the period you want to track (Options: 1y/1d/1mo, 2y, 3y/3mo, 5y/5d, 6mo, 10y, ytd, max): \n")
    data = userinput.history(period=periods)
    if data.empty:
        print("\n (AS YOU CAN TELL FROM THE ERROR) - INVALID COMMODITY SYMBOL/PERIOD, TRY AGAIN.\n")
        continue
    else:
        period_lookup = {"1d": "a day", "5d": "5 days", "1mo": "a month", "3mo": "3 months","6mo": "6 months", "1y": "a year", "2y": "2 years", "3y": "3 years", "5y": "5 years", "10y": "10 years", "ytd": "year to date", "max": "its entire lifetime"}
        clean_period_name = period_lookup.get(periods.lower(), periods)
        print(f"\n Success! Here is your commodity({userinput.ticker}) data over {clean_period_name}\n")

        avg_price = data['Close'].mean()
        daily_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
        avg_daily_return = daily_returns.mean()
        return_points = (avg_daily_return * 252) * 100
        std_dev = daily_returns.std()
        ann_volatility = std_dev * (252 ** 0.5)
        skew = daily_returns.skew()
        if pd.isna(skew): 
            skew = 0
        skew_penalty = math.copysign(1.0, skew) * (2.5 ** abs(skew)) if skew != 0 else 0
        volatility_penalty = ann_volatility * 20
        avg_volume = userinput.info.get('averageVolume', 500000)
        if avg_volume is None: 
            avg_volume = 500000
        dollar_liquidity = avg_price * avg_volume
        if dollar_liquidity >= 20000000:
            volume_penalty = 0
        elif dollar_liquidity < 5000000:
            volume_penalty = ((1.2 ** ((5000000 - dollar_liquidity) / 250000)) - 1)
        else:
            volume_penalty = 0.5
        market_beta = userinput.info.get('beta', 1.0)
        beta_penalty = 0
        if market_beta is None: 
            market_beta = 1.0 
        if market_beta > 1.2:
            steps_above = (market_beta - 1.2) / 0.2
            beta_penalty = -(steps_above * 2.0)  
        elif market_beta < 0.8:
            beta_penalty = (0.8 - market_beta) / 0.1 
        sia = SentimentIntensityAnalyzer()
        news_stories = userinput.news
        compound_scores = []
        if news_stories:
            for story in news_stories[:5]: 
                title = story.get('title', '')                     
                if not title:
                    content_block = story.get('content', {})
                    title = content_block.get('title', '') 
                if title:
                    story_score = sia.polarity_scores(title)['compound']
                    compound_scores.append(story_score)
        if compound_scores:
            avg_sentiment = sum(compound_scores) / len(compound_scores)
        else:
            avg_sentiment = 0.0 
        sentiment_score = avg_sentiment * 5.0
        score = return_points - volatility_penalty + skew_penalty - volume_penalty + beta_penalty + sentiment_score

        report_layout = f"""
        ---------------------------------------------------------------------------------------
        CUSTOM REPORT FOR : {userinput.ticker} OVER {clean_period_name.upper()}
        ---------------------------------------------------------------------------------------
        -> Average closing price:         ${avg_price:.2f}
        -> Average daily return:          {avg_daily_return * 100:.2f}%
        -> Annualized volatility:         {ann_volatility * 100:.2f}%
        -> Skewness:                      {skew:.2f}
        -> Average daily liquidity:       ${dollar_liquidity:,.2f}
        -> Beta:                          {market_beta:.2f}
        -> Average news sentiment score:  {avg_sentiment:+.2f}
        ---------------------------------------------------------------------------------------
        -> Rating Scale Criteria:
           • Less than -20:               Pack it up!
           • -20 to 0:                    Playing with fire!
           • 0 to 10:                     Okay, I see the vision!
           • 10 to 30:                    Impressive, Warren Buffett!
           • More than 30:                I'm gonna steal your commodity now!
        ---------------------------------------------------------------------------------------
        FINAL SCORE:   {score:.2f}
        ---------------------------------------------------------------------------------------
        """
        print(report_layout)

        while(True):
            user_choice = input("\nWould you like to analyze another commodity? (y/n): ").strip().lower()
            if user_choice == 'n':
                print("\nThank you for using the Commodity Analyzer! Goodbye!\n")
                running = False
                break
            elif user_choice == 'y':
                print("\nGreat! Let's analyze another commodity.\n")
                break
            else:
                print("\nInvalid input. Please enter 'y' or 'n'.\n")
                continue
        

#Printing Commodity Metrics!    
#Asking The User!
