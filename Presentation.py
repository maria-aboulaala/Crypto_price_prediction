import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

st.set_page_config(page_icon=":game_die:", page_title="Aboulaala Projet")

st.image('ensa.png',caption=None, width=250, use_column_width=None, clamp = False, channels="RGB", output_format="auto")

st.title('Projet de Séries Temporelles :bar_chart:')



with st.expander("Presentation"):

    st.markdown(
    """
> Cette presentation est faite dans le cadre du projet 
- Realiser par : Aboulaala Maria | Aberhouche Anass 
- Encadrer par : Madame Hadiri Soukaina
   """
)

st.subheader('Definition :book: :')
st.markdown(
    """
    >Les séries temporelles sont une méthode d'analyse de données qui permettent d'étudier les tendances et les modèles dans les données chronologiques. Dans un contexte de crypto-monnaie, l'analyse de séries temporelles peut être utilisée pour prédire le prix futur d'une crypto-monnaie en se basant sur les données historiques des prix et d'autres facteurs tels que la volatilité et le volume des échanges.
   """
)

st.subheader(' Presentation:')
st.markdown(
    """
    >La régression logistique est un type d'analyse statistique qui permet de prédire une variable cible catégorique (comme un gain ou une perte) en fonction d'autres variables indépendantes. Elle peut être utilisée pour l'optimisation d'un portefeuille de crypto-monnaies en prédisant la performance future des crypto-monnaies et en ajustant la répartition des actifs dans le portefeuille en conséquence.
    Pour utiliser la régression logistique pour l'optimisation d'un portefeuille de crypto-monnaies, il faut d'abord:
    - :one: Collecter des données sur les prix historiques, les volumes de négociation et d'autres indicateurs pertinents pour les crypto-monnaies que l'on souhaite inclure dans le portefeuille.
    - :two: Entraîner le modèle de régression logistique qui prédit la performance future des crypto-monnaies a partir des donnees precedente.
    - :three: Une fois que le modèle est entraîné, il peut être utilisé pour répartir les actifs dans le portefeuille de manière à maximiser les rendements et minimiser les risques.
    Par exemple, on peut utiliser les prédictions du modèle pour investir davantage dans les crypto-monnaies qui sont prévues pour performer bien, tandis que les crypto-monnaies qui sont prévues pour performer moins bien peuvent être réduites ou éliminées du portefeuille.

   """
)

st.header(':one:  ')
st.markdown(
    """
     **Definition**
    > gsgufyge
   """
)







st.subheader('Entrez les parametres: :key: ')
with st.form(key="my_form"):



    tickers = st.selectbox('Coins', ['BTC-USD', 'ETH-USD', 'XRP-USD', 'BCH-USD','LTC-USD','USDT-USD','LINK-USD','ADA-USD','DOT-USD','BSV-USD','EOS-USD','ATOM-USD','SOL-USD','CRO-USD','XMR-USD'])
    st.form_submit_button("Simuler")
st.subheader('graphique :chart:')
#display the data
data = yf.download(tickers)
dataframe = st.write(data['Close'])

with st.expander("Interpretation:"):
    st.markdown("""
    none
                """
    )

#display the chart
st.line_chart(data['Close'], use_container_width=True)

#display hist
fig, ax = plt.subplots()
ax.hist(data['Close'], bins=30)
st.pyplot(fig)

#description
info = data['Close'].describe()
df_info = pd.DataFrame(info)
st.write(df_info)

fig2 = plot_acf(data['Close'], lags=50)
plt.show()
st.pyplot(fig2)