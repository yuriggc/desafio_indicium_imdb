#Yuri Gonzaga

import os
import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from EDA import log_transform

if not os.path.exists('./report/'):
    os.makedirs('./report/')

data_filename = "desafio_indicium_imdb.csv"

#abrir csv e tratar dados
data = pd.read_csv(data_filename)
data.dropna(inplace=True)
data = data.drop(['Unnamed: 0'], axis = 1)
data['Released_Year'][965] = "1995"
data['Released_Year'] = pd.to_numeric(data['Released_Year'])
data['Runtime'] = data['Runtime'].str.split().str.get(0)
data['Runtime'] = pd.to_numeric(data['Runtime'])
data['Gross'] = data['Gross'].str.replace(',','')
data['Gross'] = pd.to_numeric(data['Gross'])

#Transformando Gross e No_of_Votes em log
log_transform(data,['Gross','No_of_Votes'])

#Obter arquivo de correlações
data.corr().to_csv('./report/corr.csv')

#Definindo features e labels
X = data[['No_of_Votes_log','Meta_score','Runtime']]
y = data['IMDB_Rating']
poly = PolynomialFeatures(degree=4, include_bias=False)
Z = poly.fit_transform(X)

#Separando os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(Z, y, test_size=0.3, random_state=10)

#Selecionando o método de regressão
model = LinearRegression().fit(X_train, y_train)

#Validação
#Avaliando o modelo a partir do treino
y_prediction = model.predict(X_train)
print("MAE on train data= " , metrics.mean_absolute_error(y_train, y_prediction)) #0.15180148694406334
#Avaliando o modelo a partir do teste
y_prediction = model.predict(X_test)
print("MAE on test data = " , metrics.mean_absolute_error(y_test, y_prediction)) #0.16584287371815784

#Predição do exemplo fornecido
ex = {'Runtime': [142], 'Meta_score': [80], 'No_of_Votes': [2343110]}
Exemplo = pd.DataFrame(data=ex)
log_transform(Exemplo,['No_of_Votes'])
Exemplo_poly = poly.fit_transform(Exemplo[['No_of_Votes_log','Meta_score','Runtime']])
model.predict(Exemplo_poly) #IMDB_Rating predito: 9.1303068

#Salvando o modelo
pkl.dump(model, open("model.pkl", 'wb'))
