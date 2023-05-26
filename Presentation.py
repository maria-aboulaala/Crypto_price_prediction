#packages
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
st.success('Notre interface interactive de prédiction du prix des cryptomonnaies repose sur l analyse des séries temporelles. Son objectif est de vous fournir des informations prédictives sur les tendances des prix des cryptomonnaies ainsi que les interpretations des resultats.')

st.title('Prédiction du prix des cryptomonnaies à l aide de l analyse de séries temporelles :bar_chart:')



with st.expander("Presentation"):

    st.markdown(
    """
> Cette presentation est faite dans le cadre du projet de prédiction du prix des cryptomonnaies à l aide de l analyse de séries temporelles
- Realiser par : Aboulaala Maria | Aberhouch Anass | Bari Said
- Encadrer par : Madame Hadiri Soukaina
   """
)

st.subheader('Definition :book: :')
st.markdown(
    """
    >Les séries temporelles sont une méthode d'analyse de données qui permettent d'étudier les tendances et les modèles dans les données chronologiques. Dans un contexte de crypto-monnaie, l'analyse de séries temporelles peut être utilisée pour prédire le prix futur d'une crypto-monnaie en se basant sur les données historiques des prix et d'autres facteurs tels que la volatilité et le volume des échanges.
   """
)

st.subheader(' Presentation: :one:')
st.markdown(
    """
    >Dans ce projet, nous nous intéressons à l'étude des données de crypto-monnaies. Les crypto-monnaies, telles que Bitcoin, Ethereum et Litecoin, ont connu une popularité croissante ces dernières années en tant qu'actifs financiers numériques. L'analyse des séries temporelles de ces crypto-monnaies peut nous aider à comprendre leurs tendances et leur volatilité, ce qui peut être utile pour les investisseurs et les traders.

    >L'objectif de ce projet est d'appliquer des techniques d'analyse de série temporelle pour modéliser et prévoir les prix des crypto-monnaies. Nous allons explorer les différentes étapes du processus d'analyse, de la préparation des données à l'estimation des modèles, en passant par le diagnostic et la validation des modèles.

   """
)



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


st.subheader('Description :chart:')
#description
info = data['Close'].describe()
df_info = pd.DataFrame(info)
st.write(df_info)

#correlograme
fig2 = plot_acf(data['Close'], lags=50)
st.pyplot(fig2)

#
resultat = adfuller(data['Close'])
#Affichage du résultat      
st.write('Statistique de test  ', resultat[0])
st.write('La P-value ', resultat[1])
st.write("Valeurs critiques :", 
         )
for key, value in resultat[4].items():
    st.write(f"\t{key}:" ,value)

#-----
st.subheader("Phillips person")
pp_test = PhillipsPerron(data['Close'])
st.write('Statistique de test :', pp_test.stat)
st.write('P-value :', pp_test.pvalue)
st.write('Valeurs critiques :')
for key, value in pp_test.critical_values.items():
    st.write(f"\t{key}:" ,value)

#-------
# Appliquer une différenciation à la série chronologique du coin
diff_data = data['Close'].diff().dropna()

# Afficher un graphique de la série chronologique différenciée
fig, ax = plt.subplots()
ax.plot(diff_data.index, diff_data, label='diff')
ax.set_title('Série chronologique différenciée')
ax.set_xlabel('jours')
ax.set_ylabel('diff')
ax.legend()
cd = plt.show()
st.pyplot(cd)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Graphique d'autocorrélations simples
plot_acf(diff_data, lags=30)
ocs = plt.show()
st.pyplot(ocs)

# Graphique d'autocorrélations partielles
plot_pacf(diff_data, lags=30)
ocp = plt.show()
st.pyplot(ocp)

#-------
import statsmodels.api as sm
import numpy as np
train_size = int(len(diff_data) * 0.8)
train, test = diff_data[:train_size], diff_data[train_size:]
p_max = 4
q_max = 4
best_aic = np.inf 
for p in range(p_max+1):
    for q in range(q_max+1):
        try:
            model = sm.tsa.arima.ARIMA(train, order=(p,1,q))
            results = model.fit()
            aic = results.aic
            if aic < best_aic:
                best_aic = aic
                best_order = (p,1,q)
                
        except:
            continue

                



#st.write('Best ARIMA{} model - AIC:{}'.format(best_order, best_aic))
st.write("Best arima model", best_order)
st.write("AIC:", best_aic)
# Ajuster le modèle ARIMA(p,q) avec les données d'entraînement
model = sm.tsa.arima.ARIMA(train, order=best_order)
results = model.fit()

# Afficher un résumé des résultats du modèle ajusté
st.write(results.summary())
#-------
residuals = pd.DataFrame(results.resid, columns=['Residuals'])
fig, ax = plt.subplots(figsize=(10, 5))
residuals.plot(ax=ax)
ax.set(title='Résidus du modèle ARIMA(2,1,4)', ylabel='Valeurs résiduelles')
z=plt.show()
st.pyplot(z)
# Tracer la fonction d'autocorrélation des résidus
fig, ax = plt.subplots(figsize=(10, 5))
plot_acf(residuals, ax=ax, lags=20)
w= plt.show()
st.pyplot(w)

#------
n_periods = len(test)
forecast = results.forecast(steps=n_periods)
st.write(forecast)

mse = ((forecast - test) ** 2).mean()
st.write("Mean Squared Error (MSE):", mse)

fig, ax = plt.subplots(figsize=(10, 5))
plt.plot(test.index, test.values, label='Données de test')
plt.plot(test.index, forecast, label='Prévisions')
ax.set(title='Prévisions avec le modèle arima')
st.pyplot(fig)



st.error('On remarque que le modele ARIMA ne donne pas de bon resultats du coup on procede au modele GARCH')