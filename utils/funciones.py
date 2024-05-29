import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris 
import seaborn as sns
from utils import variables as var
from sklearn.linear_model import LinearRegression
import yfinance as yf

def fill_null(df:pd, tkr, col, missing):
    ticker = yf.Ticker(tkr)
    df.loc[df['Symbol']==tkr, col]= ticker.info[missing]