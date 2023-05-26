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

st.subheader('Definition :book: :')
st.markdown(
    """
    >Les séries temporelles sont une méthode d'analyse de données qui permettent d'étudier les tendances et les modèles dans les données chronologiques. Dans un contexte de crypto-monnaie, l'analyse de séries temporelles peut être utilisée pour prédire le prix futur d'une crypto-monnaie en se basant sur les données historiques des prix et d'autres facteurs tels que la volatilité et le volume des échanges.
   """
)

st.subheader(' Presentation: :one:')
st.markdown(
    """
GARCH signifie **Generalized Autoregressive Conditional Heteroskedasticity**, ce qui est une extension du modèle ARCH (Autoregressive Conditional Heteroskedasticity).

GARCH comprend des termes de variance en retard avec des erreurs résiduelles en retard d'un processus moyen, et constitue l'approche économétrique traditionnelle pour la prédiction de la volatilité des séries temporelles financières.

Mathématiquement, GARCH peut être représenté de la manière suivante:

   """
)

st.latex(r"\sigma_t^2 = \omega + \sum_{i}^{q}\alpha_{i}\epsilon_{t-i}^2 + \sum_{1}^{p}\beta_{i}\sigma_{t-i}^2")


st.markdown(r"Ou $\sigma_{t}^2$ est ls variance au temps $t$ er $\epsilon_{t-i}^2$ est le rididu du modele a l'instant $t-1$")

import streamlit as st

st.markdown("""GARCH(1,1) ne contient que des termes retardés d'ordre un et son équation mathématique est la suivante :


""")

st.markdown(

r"où $\alpha$, $\beta$ et $\omega$ se somment à 1, et $\omega$ est la variance à long terme.")
st.markdown("""
Le GARCH est généralement considéré comme une amélioration significative par rapport à l'hypothèse naïve selon laquelle la volatilité future sera similaire à celle du passé, mais certains experts en volatilité le considèrent largement surestimé en tant que prédicteur. Les modèles GARCH capturent les caractéristiques essentielles de la volatilité : la volatilité de demain sera proche de celle d'aujourd'hui (**regroupement**), et la volatilité à long terme aura tendance à **revenir à la moyenne** (ce qui signifie qu'elle sera proche de la moyenne historique à long terme).""")



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

