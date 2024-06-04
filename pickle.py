# Básicos
# ==============================================================================

import pandas as pd
import numpy as np
import pickle
import datetime as dt
import scipy
from utils import funciones as fun

# Tratamiento de datos
# ==============================================================================

import statsmodels.api as sm
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import math

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import style
from pandas.plotting import autocorrelation_plot

#Financiero
import yfinance as yf
import quantstats as qs
import ta

# Preprocesado y modelado
# ==============================================================================

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, root_mean_squared_error, mean_squared_error
from prophet import Prophet
from pmdarima.arima import auto_arima
from pmdarima.arima import ARIMA
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from plotly.offline import init_notebook_mode, iplot
from plotly import graph_objs as go
from keras.models import Sequential
from keras.layers import Dense, LSTM

with open('finished_model.model', "wb") as archivo_salida:
    pickle.dump(models_gridsearch['reg_log'].best_estimator_, archivo_salida)