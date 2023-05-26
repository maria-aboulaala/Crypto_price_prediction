import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import PhillipsPerron

st.set_page_config(page_icon=":game_die:", page_title="serie temporelle")

st.image('ensa.png',caption=None, width=250, use_column_width=None, clamp = False, channels="RGB", output_format="auto")

st.title('Methode GARCH :bar_chart:')


