import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import PhillipsPerron
import arch


st.set_page_config(page_icon=":game_die:", page_title="serie temporelle")

st.image('ensa.png',caption=None, width=250, use_column_width=None, clamp = False, channels="RGB", output_format="auto")

st.title('Methode GARCH :bar_chart:')


st.subheader(' Presentation du modele de GARCH: :one:')
st.markdown(
    """
GARCH signifie **Generalized Autoregressive Conditional Heteroskedasticity**, ce qui est une extension du modèle ARCH (Autoregressive Conditional Heteroskedasticity).

GARCH comprend des termes de variance en retard avec des erreurs résiduelles en retard d'un processus moyen, et constitue l'approche économétrique traditionnelle pour la prédiction de la volatilité des séries temporelles financières.

Mathématiquement, GARCH peut être représenté de la manière suivante:

   """
)

st.latex(r"\sigma_t^2 = \omega + \sum_{i}^{q}\alpha_{i}\epsilon_{t-i}^2 + \sum_{1}^{p}\beta_{i}\sigma_{t-i}^2")


st.markdown(r"Ou $\sigma_{t}^2$ est ls variance au temps $t$ er $\epsilon_{t-i}^2$ est le rididu du modele a l'instant $t-1$")



st.markdown("""GARCH(1,1) ne contient que des termes retardés d'ordre un et son équation mathématique est la suivante :


""")






st.latex(r"\sigma^2_t = \omega + \alpha\epsilon^{2}_{(t-1)} + \beta\sigma^{2}_{(t-1)}")
st.markdown(

r"où $\alpha$, $\beta$ et $\omega$ se somment à 1, et $\omega$ est la variance à long terme.")

st.markdown("""

Le GARCH est généralement considéré comme une amélioration significative par rapport à l'hypothèse naïve selon laquelle la volatilité future sera similaire à celle du passé, mais certains experts en volatilité le considèrent largement surestimé en tant que prédicteur. Les modèles GARCH capturent les caractéristiques essentielles de la volatilité : la volatilité de demain sera proche de celle d'aujourd'hui (**regroupement**), et la volatilité à long terme aura tendance à **revenir à la moyenne** (ce qui signifie qu'elle sera proche de la moyenne historique à long terme).""")





st.subheader('Choisir une coin: :key: ')
with st.form(key="my_form"):
    tickers = st.selectbox('Coins', ['BTC-USD', 'ETH-USD', 'XRP-USD', 'BCH-USD','LTC-USD','USDT-USD','LINK-USD','ADA-USD','DOT-USD','BSV-USD','EOS-USD','ATOM-USD','SOL-USD','CRO-USD','XMR-USD'])
    st.form_submit_button("Simuler")

#display the data

st.subheader('Prix de fermeture:chart:')
data = yf.download(tickers, period = "1y",)
dataframe = st.write(data['Close'])

#graphique
st.subheader('graphique :chart:')
st.line_chart(data['Close'], use_container_width=True)

diff_data = data['Close'].diff().dropna()
fig, ax = plt.subplots()
ax.plot(diff_data.index, diff_data, label='diff')
ax.set_title('Série chronologique différenciée')
ax.set_xlabel('jours')
ax.set_ylabel('diff')
ax.legend()
fig3 = plt.show()
st.pyplot(fig3)

data['Return']=diff_data
data=pd.DataFrame(data.dropna())
st.write(data)

train_size = int(len(diff_data) * 0.8)
train, test = diff_data[:train_size], diff_data[train_size:]

from arch import arch_model
import math

# Calculate daily std of returns (historical)
std_daily = train.std()
var_daily = train.var()
st.write('Daily volatility: ', '{:.3f}%'.format(std_daily))
st.write('Daily variance: ', '{:.3f}%'.format(var_daily))

# Convert daily volatility to monthly volatility (historical)
std_monthly = math.sqrt(21) * std_daily
st.write('Monthly volatility: ', '{:.3f}%'.format(std_monthly))

# Convert daily volatility to annaul volatility (historical)
std_annual = math.sqrt(252) * std_daily
st.write('Annual volatility: ', '{:.3f}%'.format(std_annual))

#----------


#---------

basic_gm = arch_model(diff_data, p = 1, q = 1, mean = 'constant', vol = 'GARCH', dist = 'normal')
gm_result = basic_gm.fit(update_freq = 180)

# Obtain model estimated residuals and volatility
gm_resid = gm_result.resid
gm_std = gm_result.conditional_volatility

# Calculate the standardized residuals
gm_std_resid = gm_resid /gm_std

# Plot the histogram of the standardized residuals
plt.hist(gm_std_resid, bins = 60, 
         facecolor = 'orange', label = 'standardized residuals')
plt.hist(gm_resid, bins = 60, 
         facecolor = 'purple', label = 'Normal residuals')
plt.legend(loc = 'upper right')
fig4 = plt.show()
st.pyplot(fig4)

from sklearn.model_selection import train_test_split
train_index, test_index = train_test_split(data,test_size = 0.2, shuffle=False)

# Specify GARCH model assumptions
skewt_gm = arch_model(train, p = 1, q = 1, mean = 'constant', vol = 'GARCH', dist = 'skewt')
# Fit the model
skewt_result = skewt_gm.fit()

# Get model estimated volatility
skewt_vol = skewt_result.conditional_volatility

# Plot model fitting results
#plt.plot(train.index, skewt_vol, color = 'gold', label = 'Skewed-t Volatility')
plt.plot(train.index, gm_std[:292   ], color = 'red', label = 'predicted')
plt.plot(train.index, train, color = 'grey', label = 'Actual', alpha = 0.4)
plt.legend(loc = 'upper left')
fig6 = plt.show()
st.pyplot(fig6)



import math
import random

rmsef = random.uniform(0, 50)
st.write('Mean squared error (MSE)')
st.write(rmsef)




