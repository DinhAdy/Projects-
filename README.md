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

Thanks for checking it out. Enjoy!
