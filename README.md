Welcome! Here, you will find a collection of my project so far, as well as brief descriptions of them. Feel free to take a look!

                                                   Commodity.py 
Commodity.py is a fun little side project I made. It pulls stock data in a flash, calculates core statistical variance metrics, then uses my custom scoring system to rate it.

I've made the code as concise and as user friendly as possible. Be warned that for this, the API doesn't work on Python 3.14+. It has to be 3.13- because it's a stabilized version of Python. Other than that, it automatically installs the required "Yfinance" library on your computer if you don't have it. Think of it as a token of gratitude for checking out my code. Now you have access to a really cool finance API :]

Occasionally, I'll upgrade the version of the file. 

Version 1.1.0, has 5 core factor that it measures

-> Average Daily Return

-> Annualized Volatility

-> Skewness

-> Daily Trading Volume

-> Market Beta

Version 1.2.0 has a couple of upgrades

-> Average Daily Return is upgraded so I take the natural logarithm of it, accounting for market behaviour

-> Daily Trading Volume is now upgraded to Daily Liquidity by multiplying volume with share price, making it more accurate

-> Cleaned up the formatting so its less overly-descriptive, gets straight to the point and is cleaner

-> Added 6 months as a period option(Sorry! Missed it in version 1.1.0)

-> Allows for infinite requests(one at a time) instead of ending the program after a commodity is analyzed. You can exit the program by pressing the prompt after a commodity is analyzed

-> Added in a new factor "sentiment score" that uses an external free API to do a quick glance at the news and rate it

                                                  SP500entirething.py

SP500EntireThing.py is a fun little deviation of Commodity.py. It scales my project up to the S&P 500, which basically tracks the entire US Market. Some cool features:

-> It has every single one of the metrics calculated in Commodity.py and scores it just like Commodity.py except for news. NEWS FLASH(pun intended), searching up roughly 500 different stock's news articles, parsing through them and scoring them using AI is extremely tedious and time consuming! As such, for practical and balancing purposes, I have removed that feature.

-> It also features custom periods input and custom multithreading depending on how strong your CPU is, to make it run as fast as your computer can make it run! Other than that, it does some fancy matrix stuff in the background, runs metrics for 500 stocks and just prints them out for you! 

-> If I feel like it in the future, I'll add a feature where you guys can customize the amount of top-ranked stocks it prints out. But for now, peace!


Version 4.0.0 changes:


-> Added a batched Monte Carlo simulation using a Cornish-Fisher expansion optimized via Horner's method. It calculates a simple 95% 1-day Value-at-Risk to heavily penalize assets with asymmetric crash risk.


-> Added exponentially weighted moments with a 63-day half-life so recent market volatility, skewness, and kurtosis carry more weight in the risk score than older data.


-> Replaced price/volume columns with CF_Downside_Risk_% and Excess_Kurtosis (measuring fat-tail crash risk).


-> Completely removed average price, volume tracking, and volume penalty, shifting penalty weighting directly to the Monte Carlo downside risk.


-> Batches the S&P 500 stocks in neat 50-stock display batches, reports total scanned universe size, and replaces missing/NaN rows cleanly with "N/A".


-> Bypasses the manual loop prompt and executes automatically with a default 1-year lookback (period="1y").


-> Added importlib and ensurepip fallback bootstrapping for strict Python environments, alongside encoding changes to prevent Windows terminal character crashes.



                                                    montecarlo.py


This project is a high-speed Python tool that backtests custom stock portfolios and simulates 14 million imaginary future market paths to check total risk exposure. The entire process takes under one second (907ms) and uses a tiny 326 MiB of memory.


First off, to get it out of the way! I do not have a degree in advanced financial mathematics, and I did not write every single line of this code by hand.
I used a two-step AI assembly line to build this framework.

Second! The explanations below are simplified for intuition. They are not rigorous academic definitions, but rather conceptual analogies to explain what the code is doing.


My Personal Code: The one part I manually built entirely from my own experience is the dynamic computer core and thread-handling safety block. I figured out how to safely balance processor threads by optimizing modded Minecraft Java servers when I was a kid so my laptop wouldn't freeze up under a heavy load. I just applied that exact same hardware logic here to let the script download five years of stock data over the internet in a few seconds.


Core Features


Simple User Setup: You just paste your hand-picked stock list directly into the code array. An auto-installer at the top automatically scans your laptop, grabs Python's tool installer, and downloads all required libraries with zero manual setup.


Real-World Friction: Unlike simple textbook simulators, this backtester forces the strategy to pay real-world trading penalties day-by-day—including broker commissions, market execution slippage, and sell-side government SEC transaction taxes.


Regime Crash Testing: If the VIX market fear index spikes past a high threshold (VIX > 25), the script automatically triggers a shock multiplier to individual stock risks, forcing the system to actively defend against market panics.


Total Execution Time: 907 milliseconds(Lowest time recorded. Can vary quite a bit but virtually should never exceed 1100ms)


Peak Estimated Memory Ceiling: 325.98 MiB RAM (Fits entirely within local CPU cache memory to maximize processing velocity).


Friction-Adjusted Scorecard: For this particular portfolio, which I selected by handpicking stocks from SP500entirething.py, model says I got a +96.57% Net Realized Return over a 1,001-day historical timeline, 1.007 Annual Sharpe Ratio and holding the max crash dip to -13.72%. Although in my opinion, I kind of dislike the Sharpe Ratio, since it treats volatility as bad when volatility is just a double-edged sword! It can be bad or good, but its not inherently bad like the ratio says it is. If I have some more time to dedicate to this, I might bump it up to a Sortino Ratio, which is better.


                                                      Mc.py


This project MC(short for Monte Carlo), is an upgraded version of my old Monte Carlo project(montecarlo.py) ! It has a bunch of new snazzy upgrades, and finally, the Sortino upgrade I promised you guys earlier :)

First off, the explanations below are simplified for intuition. They are not rigorous academic definitions, but rather conceptual analogies to explain what the code is doing.


- Multi-Factor Risk Modeling: Integrated 7 macro factor proxies (Market, Size, Value, Rates, Gold, Healthcare, Energy) alongside live 13-week T-Bill risk-free rates.


Explanation: This simply takes into account different sectors and bonds in the US market.

  
- Regime-Switching Simulation Engine: Implemented a 3-state (Calm, High-Vol, Panic) VIX transition model featuring semi-parametric fat-tail augmentation for extreme crash modeling.


Explanation: This simply tells you the state of the market and adjusts the metrics accordingly
  
- Euler Risk Decomposition: Calculated marginal asset betas and percentage tail-risk contributions to identify portfolio "Tail Aggressors" vs. "Diversifiers". 


Explanation: This simply tells you which stock is driving the portfolio down the most during a crash and which one is dragging the portfolio up/padding the loss.

  
- Liquidity-Adjusted VaR (L-VaR): Incorporated Amihud illiquidity metrics to simulate fire-sale liquidation haircuts and dynamic trade slippage under market stress.


Explanation: This simply penalizes very niche stocks due to the fact that they are incredibly risky to attempt to execute buy/sell orders. 
  
- Ledoit-Wolf Shrinkage & FHS: Re-calibrated regime-conditioned covariance matrices using Ledoit-Wolf shrinkage combined with Filtered Historical Simulation.


Explanation: When you calculate stock data using matrices, if you just dump raw data into it, there's noise that causes all sorts of weird stuff. So Ledoit-Wolf Shrinkage and FHS is a way to trim the fat. For example, you bet for the 2026 World Cup that Argentina is going to win against Algeria because the data says so statistically. But let's say Algeria has a promising rookie and they just got a really good goalie. I'm not saying the Argentina historical datas don't matter, but they're only a part of the picture. You trim the fat on the bloated data, and introduce new factors, like checking for panics and industry correlation.


Again, this was wrote using the same old asssembly AI workflow! The lowest runtime was bumped to around 6.7(HAHA) seconds, but that's because I rolled it over a 10 day horizon rather than a 1 day horizon, so it was inevitable. However, the peak usage of RAM stayed roughly the same. So yeah, a bunch of new cool fancy advanced metrics, as well as that Sortino bump and a longer period of analysis.

                                              MC(1.1.0).py


Once again, the explanations below are simplified for intuition. They are not rigorous academic definitions, but rather conceptual analogies to explain what the code is doing.


Upgraded the past Monte Carlo by adding in:


GJR-GARCH Path Memory: Unlike simple Monte Carlo sims, this engine remembers volatility shocks. If the market crashes on Day 2, Day 3 is mathematically forced to react.


Volatility shock refers to the simulation taking into account the "derivative" of the volatility change with respect to time for the future data. The only difference is we do not have an exact graph of the data with infinite data points, but we have an approximation of the curve with a lot of data points(each day's data). So that is where it differs from Newtonian and Leibniz calculus.


Hybrid Rebalancing: The backtester rebalances on a 21-day schedule, a 20% weight drift, or a VIX panic override. (In this run: 105 rebalances, including 61 VIX overrides). 


This simply refers to a dynamic frequency of trading, meaning that the frequency changes depending on how the volatility is doing.


Liquidity-Adjusted VaR: Calculates fire-sale slippage based on the crashed terminal portfolio value, not the starting value. 

This simply penalizes very niche stocks due to the fact that they are incredibly risky to attempt to execute buy/sell orders. I simply changed the method in which you penalize low liquidity from MC.py.


Again, just so we're clear! I built this using an AI assembly line. Also, I beat Dr. Strange 9,194,716 times(using the Time Stone definitely no pun intended).


Thanks for checking it out. Enjoy!
