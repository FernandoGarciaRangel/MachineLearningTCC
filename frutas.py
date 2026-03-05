# %%

import pandas as pd

df = pd.read_excel("data/dados_frutas.xlsx")
df.head()

#%%

from sklearn import tree

arvore = tree.DecisionTreeClassifier()


# %%
y = df['Fruta']

caracteristicas = ['Arredondada', 'Suculenta', 'Vermelha', 'Doce']
x = df[caracteristicas]

x
# %%
