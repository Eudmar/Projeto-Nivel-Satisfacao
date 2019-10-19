## Bem Vindo

### Nivel de Satisfação dos Clientes

A satisfação do cliente é uma medida fundamental de sucesso. Clientes
insatisfeitos cancelam seus serviços e raramente expressam sua insatisfação antes
de sair. Clientes satisfeitos, por outro lado, se tornam defensores da marca!
 
Neste projeto de aprendizado de máquina, o objetivo é prever se um cliente está satisfeito ou insatisfeito com
sua experiência bancária.

Para este projeto, os dados estão disponíveis no [Kaggle](https://www.kaggle.com/c/santander-customer-satisfaction)

## Importando as bibliotecas

```markdown
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
```
