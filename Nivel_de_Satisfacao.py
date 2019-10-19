
# coding: utf-8

# ## Nivel de Satisfação dos Clientes

# A satisfação do cliente é uma medida fundamental de sucesso. Clientes
# insatisfeitos cancelam seus serviços e raramente expressam sua insatisfação antes
# de sair. Clientes satisfeitos, por outro lado, se tornam defensores da marca!
# 
# Neste projeto de aprendizado de máquina, o objetivo é prever se um cliente está satisfeito ou insatisfeito com
# sua experiência bancária.
# 
# Para este projeto, os dados estão disponíveis no Kaggle em: 
# https://www.kaggle.com/c/santander-customer-satisfaction

# In[1]:


# Imports
# manipulação de dados
import pandas as pd
import numpy as np
# gráficos
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
# eliminação recursiva de atributos
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# Preprocessamento
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# Cross Validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# Modelo XGBoost
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
# evitar avisos de warnings
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Importando dados de treino
treino = pd.read_csv("../input/santander-customer-satisfaction/train.csv")
teste = pd.read_csv("../input/santander-customer-satisfaction/test.csv")


# ****Análise Exploratória****

# In[3]:


# Observando as primeiras linhas
treino.head()


# In[4]:


# Tipo dos dados
treino.dtypes


# In[5]:


# Tamanho do banco de dados treino
len(treino)


# In[6]:


# Resumo dos dados de treino
treino.describe()


# In[7]:


# Verificando a proporção da variável TARGET
# 0 = Clientes satisfeitos e 1 = Clientes insatisfeitos
df = pd.DataFrame(treino.TARGET.value_counts())
df['Prop'] = 100 * df['TARGET'] / treino.shape[0]
df['Prop'].plot(kind = 'bar', title = 'Proporção (Target)', color = ['#1F77B4', '#FF7F0E']);


# Olhando para figura acima, fica fácil perceber que o banco de dados está desbalanciado em relação a variável TARGET.

# In[8]:


# É possível observar o valor -999999, o que caracteriza um valor missing
treino.var3.value_counts()


# In[9]:


# Substituindo os -999999 por 2 e inserindo ao banco de dados
var3_1 = treino.var3.replace(-999999, 2)
treino.insert(2, 'var3_1', var3_1)


# In[10]:


treino.var3_1.value_counts()


# In[11]:


treino.describe()


# In[12]:


# Histograma da variável var15 - possivelmente é a idade de cada cliente
sns.distplot(treino.var15, fit = stats.norm);


# In[13]:


# Boxplot
sns.boxplot(x = "TARGET", y = "var15", data = treino);


# In[14]:


# variável TARGET pela var15
sns.stripplot(x = "TARGET", y = "var15", data = treino, jitter = True);


# **Balanciando os Dados**
# 
# Método UNDERSAMPLING
# 
# Antes de fazer uma seleção das variáveis mais importantes para o modelo,
# será feito o balanciamento dos dados pelo método UNDERSAMPLING. Técnica
# bastante utilizada, na qual iguala a categoria mais alta em relação à mais baixa da 
# variável target.

# In[15]:


# Contar Classes
conte_classe_0, conte_classe_1 = treino.TARGET.value_counts()

# Dividindo por classe
df_classe_0 = treino[treino['TARGET'] == 0]
df_classe_1 = treino[treino['TARGET'] == 1]
df_classe_0_UnderS = df_classe_0.sample(conte_classe_1)
df_treino_UnderS = pd.concat([df_classe_0_UnderS, df_classe_1], axis = 0)

# Mostrando como ficou a contagem da variável TARGET
print('Under Sampling Aleatório:')
print(df_treino_UnderS.TARGET.value_counts())
df_treino_UnderS.TARGET.value_counts().plot(kind = 'bar', title = 'Contagem (target)', color = ['#1F77B4', '#FF7F0E']);


# ****Eliminação Recursiva de Atributos****

# In[16]:


# Separando a variável TARGET
array = df_treino_UnderS.values

X = array[:,1:371]
Y = array[:,371]

# Semente

seed = 123

# Criação do modelo
modelo = LogisticRegression(random_state = seed)

# RFE - Eliminação Recursiva de Atributos
rfe = RFE(modelo, 10) # Os 10 mais importantes
fit = rfe.fit(X, Y)

# Print dos resultados
print('Variáveis Preditoras:', treino.columns[1:371])
print('Variáveis Selecionadas: %s' % fit.support_)
# o número 1 são as variáveis que apresentaram melhor resultado
print('Ranking dos Atributos: %s' % fit.ranking_)
print('Número de Melhores Atributos: %d' % fit.n_features_)


# In[17]:


# Variáveis selecionadas

var_selec_treino = df_treino_UnderS[['ind_var30', 'ind_var30_0','num_var5','num_var8_0','num_var13_0', 'num_var13_corto_0', 'num_var13_corto',
                              'num_var42', 'num_meses_var5_ult3', 'num_meses_var8_ult3']]


# In[18]:


# Transformando os dados
# Dados
X_treino = var_selec_treino

# Gerando a nova escala (normalizando os dados entre 0 e 1)
X_treino = MinMaxScaler().fit_transform(X_treino)

# Padronizando os dados
X_treino = StandardScaler().fit_transform(X_treino)


# In[19]:


# Carregando os dados

Y_treino = df_treino_UnderS['TARGET'].values

# Definindo os valores para o número de folds
num_folds = 10
seed = 7

# Separando os dados em folds
kfold = KFold(num_folds, True, random_state = seed)

# Criando o modelo
modelo = XGBClassifier().fit(X_treino, Y_treino)

# Cross Validation
resultado = cross_val_score(modelo, X_treino, Y_treino, cv = kfold)


# In[20]:


#Variáveis de teste selecionadas

var_selec_teste = teste[['ind_var30', 'ind_var30_0','num_var5','num_var8_0','num_var13_0', 'num_var13_corto_0', 'num_var13_corto',
                              'num_var42', 'num_meses_var5_ult3', 'num_meses_var8_ult3']]


# In[21]:


# Dados de teste
X_test = var_selec_teste

# Normalizando os dados
X_test = MinMaxScaler().fit_transform(X_test)

# Padronizando os dados
X_test = StandardScaler().fit_transform(X_test)

# Previsões com os dados de teste
pred = modelo.predict(X_test)


# In[22]:


# Salvando os dados da previsão
previsao = pd.DataFrame()
previsao['ID'] = teste["ID"]
previsao['TARGET'] = pred

previsao.to_csv('previsao.csv', index = False)


# In[23]:


print(previsao)

