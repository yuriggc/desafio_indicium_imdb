#Yuri Gonzaga

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def log_transform(data,col):
        for colname in col:
                if (data[colname] == 1.0).all():
                    data[colname + '_log'] = np.log(data[colname]+1)
                else:
                    data[colname + '_log'] = np.log(data[colname])

def main():
        folders = ['./report/','./plot/','./plot/num/','./plot/cat/','./plot/log/','./plot/vsGross/','./plot/vsIMDB_Rating/']
        for f in folders:
                if not os.path.exists(f): 
                        os.makedirs(f)

        data_filename = "desafio_indicium_imdb.csv"

        #abrir csv e fazer primeiras observações
        data = pd.read_csv(data_filename)
        data.shape # 999 linhas e 16 colunas
        data.head()
        data.tail()
        data.info()
        data = data.drop(['Unnamed: 0'], axis = 1)

        #inserindo o ano de lançamento de Apollo 13, que estava faltando
        data['Released_Year'][965] = "1995"

        #convertendo algumas colunas para valores numéricos
        data['Released_Year'] = pd.to_numeric(data['Released_Year'])
        data['Runtime'] = data['Runtime'].str.split().str.get(0)
        data['Runtime'] = pd.to_numeric(data['Runtime'])
        data['Gross'] = data['Gross'].str.replace(',','')
        data['Gross'] = pd.to_numeric(data['Gross'])

        #quantidade de valores únicos
        data.nunique()

        #quantidade de valores vazios
        data.isnull().sum()
        #Certificate, Meta_score e Gross possuem valores faltantes

        #quebrar a coluna Genre em 3 colunas separadas
        genre1 = []
        genre2 = []
        genre3 = []
        for g in data['Genre']:
                s = g.replace(',','').split()
                t = len(s)
                genre1.append(s[0])
                if t > 1:
                        genre2.append(s[1])
                else:
                        genre2.append('')
                if t == 3:
                        genre3.append(s[2])
                else:
                        genre3.append('')
        data['Genre1'] = genre1
        data['Genre2'] = genre2
        data['Genre3'] = genre3
        data = data.drop(['Genre'], axis = 1)

        #Estatísticas
        data.describe().to_csv("./report/stats.csv")

        #Gráficos de 1 variável

        #Variáveis numéricas
        num_cols = ['Released_Year', 'Runtime', 'IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Gross']
        for col in num_cols:
                print(col)
                print('Skew :', round(data[col].skew(), 2))
                plt.figure(figsize = (15, 4))
                plt.subplot(1, 2, 1)
                data[col].hist(grid=False)
                plt.ylabel('count')
                plt.subplot(1, 2, 2)
                sns.boxplot(x=data[col])
                plt.savefig("./plot/num/"+col)
                plt.close("all")

        #Variáveis categorizadas
        fig, ax = plt.subplots(1, 1, figsize = (8, 4))
        sns.countplot(x = 'Certificate', data = data, color = 'blue', order = data['Certificate'].value_counts().index)
        ax.tick_params(labelrotation=45)
        plt.subplots_adjust(bottom=0.25)
        plt.savefig('./plot/cat/Certificate')
        plt.close("all")

        #Juntar Genre1, 2 e 3 num só gráfico
        genre = Counter(pd.concat([data['Genre1'],data['Genre2'],data['Genre3']]))
        del genre[''] #removendo as ocorrência de gênero vazio
        genre = pd.DataFrame(genre.most_common(), columns = ['Genre', 'Count'])
        plt.bar(genre['Genre'], genre['Count'])
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.25)
        plt.savefig('./plot/cat/Genre')
        plt.close("all")
            
        #Plotar apenas os TOP20 Diretores
        director = Counter(data['Director'])
        director = pd.DataFrame(director.most_common(20), columns = ['Name', 'Count'])
        plt.bar(director['Name'], director['Count'])
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.3)
        plt.savefig('./plot/cat/Director')
        plt.close("all")

        #Juntar Stars1, 2, 3 e 4 e plotar o TOP20
        stars = Counter(pd.concat([data['Star1'],data['Star2'],data['Star3'],data['Star4']]))
        stars = pd.DataFrame(stars.most_common(20), columns = ['Name', 'Count'])
        plt.bar(stars['Name'], stars['Count'])
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.3)
        plt.savefig('./plot/cat/Stars')
        plt.close("all")

        #Transformando Gross e No_of_Votes em log
        log_transform(data,['Gross','No_of_Votes'])

        sns.distplot(data['Gross_log'], axlabel="Gross_log")
        plt.savefig('./plot/log/Gross_log')
        plt.close("all")

        sns.distplot(data['No_of_Votes_log'], axlabel="No_of_Votes_log")
        plt.savefig('./plot/log/No_of_Votes_log')
        plt.close("all")

        #Gráficos de 2 variáveis
        cols = ['Certificate','Genre1','Genre2','Genre3','Director','Star1','Star2','Star3','Star4','Series_Title','Overview','No_of_Votes','Gross']

        #Para variáveis numéricas
        plt.figure(figsize=(13,17))
        sns.pairplot(data=data.drop(cols,axis=1))
        plt.savefig('./plot/numpairplot')
        plt.close("all")

        #Para variáveis categorizadas
        cols.remove('Series_Title')
        cols.remove('Overview')
        cols.remove('No_of_Votes')
        cols.remove('Gross')

        #vs Gross
        for c in cols:
                t = data[c].nunique()
                if t > 50:
                        t = 50
                data.groupby(c)['Gross_log'].mean().sort_values(ascending=False)[:t].plot.bar()
                plt.subplots_adjust(bottom=0.5)
                plt.savefig('./plot/vsGross/'+ c + ' vs Gross')
                plt.close("all")

        #vs IMDB_Rating
        for c in cols:
                t = data[c].nunique()
                if t > 50:
                        t = 50
                data.groupby(c)['IMDB_Rating'].mean().sort_values(ascending=False)[:t].plot.bar()
                plt.subplots_adjust(bottom=0.5)
                plt.savefig('./plot/vsIMDB_Rating/'+ c + ' vs IMDB_Rating')
                plt.close("all")

        #Gráfico de várias variáveis
        plt.figure(figsize=(10, 10))
        sns.heatmap(data.drop(['Gross','No_of_Votes'],axis=1).corr(), annot = True, vmin = -1, vmax = 1)
        plt.xticks(fontsize=14, rotation=90)
        plt.yticks(fontsize=14, rotation=0)
        plt.subplots_adjust(bottom=0.25, left=0.25)
        plt.savefig('./plot/heatmap')
        plt.close("all")

if __name__ == "__main__":
        main()
