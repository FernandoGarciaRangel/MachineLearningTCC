# Aprendendo Machine Learning — Parte 3: O Código Passo a Passo

> Este documento percorre cada célula do notebook `estudo_corrosao.ipynb`, explicando **o que faz**, **por que faz** e **como funciona**.

---

## Célula 1 — Importar Bibliotecas

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
```

### Linha por linha:

| Linha | O que faz |
|-------|-----------|
| `import pandas as pd` | Carrega o pandas (manipulação de tabelas). O `as pd` é um apelido — em vez de escrever `pandas.read_excel()`, escrevemos `pd.read_excel()` |
| `import numpy as np` | Carrega o numpy (operações matemáticas com arrays). Usado para gerar números aleatórios e cálculos |
| `import matplotlib.pyplot as plt` | Carrega o matplotlib (criação de gráficos). `plt` é o módulo principal para plotar |
| `import seaborn as sns` | Carrega o seaborn (gráficos estatísticos mais bonitos, construído em cima do matplotlib) |
| `from sklearn.model_selection import cross_val_score, KFold` | Importa funções de validação cruzada do scikit-learn |
| `from sklearn.linear_model import LinearRegression` | Importa o modelo de Regressão Linear |
| `from sklearn.ensemble import RandomForestRegressor` | Importa o modelo Random Forest para regressão |
| `from xgboost import XGBRegressor` | Importa o modelo XGBoost para regressão |
| `from sklearn.metrics import ...` | Importa funções para calcular métricas de erro |
| `from sklearn.preprocessing import LabelEncoder` | Importa ferramenta para converter texto em números |
| `warnings.filterwarnings('ignore')` | Esconde avisos que poluem a saída (não afeta o código) |
| `sns.set_style('whitegrid')` | Define o estilo visual dos gráficos (fundo branco com grade) |
| `plt.rcParams['figure.dpi'] = 150` | Aumenta a resolução dos gráficos (mais nítidos) |

---

## Célula 2 — Carregar os Dados do Excel

```python
raw = pd.read_excel('exemplo passagem de pig.xlsx')

colunas = raw.iloc[0].values
df = raw.iloc[1:].copy()
df.columns = [
    'id_tubo', 'posicao_m', 'dist_sold_ant_m', 'compr_tubo_m',
    'i_e', 'tipo', 'posicao_horaria', 'espessura_mm',
    'comprimento_mm', 'largura_mm', 'profundidade_pct', 'erf', 'tipo_pof'
]
```

### O que está acontecendo:

**1. `pd.read_excel('exemplo passagem de pig.xlsx')`**
- Lê o arquivo Excel inteiro e cria uma tabela (DataFrame)
- O resultado fica na variável `raw` (dados "crus", sem tratamento)

**2. `raw.iloc[0].values`**
- `iloc[0]` = "pega a linha de índice 0" (primeira linha)
- `.values` = converte para array simples
- Salvamos em `colunas` porque a primeira linha contém os nomes reais das colunas

**3. `raw.iloc[1:].copy()`**
- `iloc[1:]` = "pega da linha 1 em diante" (pula a primeira, que era o cabeçalho)
- `.copy()` = cria uma cópia independente (para não alterar `raw` acidentalmente)

**4. `df.columns = [...]`**
- Renomeia as colunas para nomes padronizados (sem acento, sem espaço, snake_case)
- Isso facilita o uso no código: `df['profundidade_pct']` é muito melhor que `df['Prof.\n(%)']`

---

## Célula 3 — Limpar e Converter os Dados

```python
def limpar_numero(valor):
    """Converte string com formato brasileiro (1.234,56) para float."""
    if isinstance(valor, str):
        return float(valor.replace('.', '').replace(',', '.'))
    return float(valor)

colunas_numericas = [
    'posicao_m', 'dist_sold_ant_m', 'compr_tubo_m',
    'espessura_mm', 'comprimento_mm', 'largura_mm',
    'profundidade_pct', 'erf'
]

for col in colunas_numericas:
    df[col] = df[col].apply(limpar_numero)

df['i_e'] = df['i_e'].replace('-', 'E')
df.reset_index(drop=True, inplace=True)
```

### Explicando a função `limpar_numero`:

O problema: no Excel brasileiro, "1.234,56" significa mil duzentos e trinta e quatro vírgula cinquenta e seis. Mas o Python lê "1.234,56" como texto, não como número.

```
Entrada: "1.234,56"
Passo 1: replace('.', '')  → "1234,56"     (remove ponto de milhar)
Passo 2: replace(',', '.') → "1234.56"     (troca vírgula por ponto decimal)
Passo 3: float(...)        → 1234.56       (converte para número)
```

### Explicando o `for` + `.apply()`:

```python
for col in colunas_numericas:      # Para cada coluna numérica...
    df[col] = df[col].apply(limpar_numero)  # Aplica a função em cada valor
```

O `.apply(funcao)` executa a função em **cada célula** da coluna. É como dizer: "para cada valor nesta coluna, passe pela função de limpeza".

### Outras linhas:

| Linha | O que faz |
|-------|-----------|
| `df['i_e'].replace('-', 'E')` | Substitui "-" por "E" (Externa) na coluna i_e |
| `df.reset_index(drop=True, inplace=True)` | Redefine o índice para começar de 0. `drop=True` = não guarda o índice antigo. `inplace=True` = modifica o df diretamente |

---

## Célula 4 — Estatísticas Descritivas

```python
df[colunas_numericas].describe().round(2)
```

- `df[colunas_numericas]` = seleciona apenas as colunas numéricas
- `.describe()` = calcula automaticamente: contagem, média, desvio padrão, mín, quartis, máx
- `.round(2)` = arredonda para 2 casas decimais

---

## Célula 5 — Gráficos de Distribuição (Histogramas)

```python
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

for ax, var, title in zip(axes.flat, vars_plot, titles):
    sns.histplot(df[var], kde=True, ax=ax, color='steelblue', edgecolor='white')
    ax.set_title(title)
```

### Explicando:

| Trecho | O que faz |
|--------|-----------|
| `plt.subplots(2, 3)` | Cria uma grade de gráficos: 2 linhas × 3 colunas = 6 gráficos |
| `figsize=(14, 8)` | Tamanho da figura em polegadas (largura, altura) |
| `axes.flat` | Transforma a grade 2×3 em uma lista plana de 6 slots |
| `zip(axes.flat, vars_plot, titles)` | Conecta cada slot com uma variável e um título |
| `sns.histplot(...)` | Cria um histograma (gráfico de barras mostrando a distribuição) |
| `kde=True` | Adiciona uma curva suave por cima (Kernel Density Estimation) |
| `ax=ax` | Diz em qual slot desenhar |

**O que é um histograma?** Divide os valores em faixas e conta quantas anomalias caem em cada faixa. Ex: quantas anomalias têm profundidade entre 10-15%, entre 15-20%, etc.

---

## Célula 6 — Mapa de Correlação

```python
sns.heatmap(df[cols_corr].corr(), annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, square=True)
```

| Trecho | O que faz |
|--------|-----------|
| `.corr()` | Calcula a correlação entre todas as colunas (de -1 a +1) |
| `sns.heatmap(...)` | Desenha a matriz como um mapa de calor colorido |
| `annot=True` | Mostra os números em cada célula |
| `cmap='coolwarm'` | Azul = correlação negativa, Vermelho = positiva |
| `center=0` | O branco fica no zero (sem correlação) |

**O que é correlação?**
- **+1.0**: Quando uma sobe, a outra sobe igual (relação perfeita)
- **0.0**: Não há relação
- **-1.0**: Quando uma sobe, a outra desce

---

## Célula 7 — Simulação de Crescimento

```python
np.random.seed(42)

taxa_base = {'CORR': 0.8, 'ASCI': 0.4, 'COSC': 0.3}

for idx, row in df.iterrows():
    # Para cada anomalia...
    for i in range(len(anos_inspecao) - 1):
        # Para cada intervalo de 5 anos...
        ruido = np.random.normal(1.0, 0.15)
        taxa_real = base * fator_prof * fator_esp * ruido
        taxa_real = max(taxa_real, 0.05)
        prof_futura = min(prof_atual + taxa_real * 5, 95)
```

### Explicando passo a passo:

**`np.random.seed(42)`** — Define a semente do gerador aleatório. Isso garante que, toda vez que rodar o código, os números "aleatórios" serão os mesmos. O 42 é arbitrário (referência ao Guia do Mochileiro das Galáxias).

**`df.iterrows()`** — Percorre o DataFrame linha por linha. Cada `row` é uma anomalia.

**`np.random.normal(1.0, 0.15)`** — Gera um número aleatório com distribuição normal (Gaussiana):
- Média = 1.0 (centralizado em 1)
- Desvio padrão = 0.15 (variação de ±15%)
- Simula a variabilidade natural do processo de corrosão

**`max(taxa_real, 0.05)`** — Garante que a taxa nunca fique abaixo de 0.05 (o ruído poderia gerar valores negativos).

**`min(prof_atual + crescimento, 95)`** — Limita a profundidade máxima em 95% (uma anomalia não pode ter mais de 100% da espessura da parede).

---

## Célula 8 — Codificar Categorias e Definir X e y

```python
le_tipo = LabelEncoder()
df_ml['tipo_cod'] = le_tipo.fit_transform(df_ml['tipo'])

features = [
    'espessura_mm', 'comprimento_mm', 'largura_mm',
    'profundidade_inicial_pct', 'erf',
    'tipo_cod', 'tipo_pof_cod', 'i_e_cod', 'ano_inicio'
]
target = 'taxa_corrosao_pct_ano'

X = df_ml[features]
y = df_ml[target]
```

### Explicando:

**`LabelEncoder()`** — Cria um "tradutor" de texto para números.

**`le_tipo.fit_transform(df_ml['tipo'])`** — Faz duas coisas:
- `fit` = "aprenda quais categorias existem" (ASCI, COSC, CORR)
- `transform` = "converta cada valor para número" (ASCI→0, CORR→1, COSC→2)

**`X = df_ml[features]`** — Seleciona apenas as 9 colunas de entrada. X é uma tabela com 846 linhas × 9 colunas.

**`y = df_ml[target]`** — Seleciona a coluna alvo. y é um vetor com 846 valores.

---

## Célula 9 — Treinar e Comparar Modelos

```python
modelos = {
    'Regressão Linear': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for nome, modelo in modelos.items():
    scores_r2 = cross_val_score(modelo, X, y, cv=kf, scoring='r2')
    scores_mae = -cross_val_score(modelo, X, y, cv=kf, scoring='neg_mean_absolute_error')
    scores_rmse = np.sqrt(-cross_val_score(modelo, X, y, cv=kf, scoring='neg_mean_squared_error'))
```

### Explicando:

**`modelos = { ... }`** — Um dicionário que mapeia o nome do modelo ao objeto do modelo. Isso permite iterar sobre todos com um `for`.

**`KFold(n_splits=5, shuffle=True, random_state=42)`** — Configura a validação cruzada:
- `n_splits=5` = divide em 5 partes
- `shuffle=True` = embaralha os dados antes de dividir (importante para evitar viés)
- `random_state=42` = garante reprodutibilidade

**`cross_val_score(modelo, X, y, cv=kf, scoring='r2')`** — Faz TUDO automaticamente:
1. Divide os dados em 5 partes
2. Para cada rodada: treina com 4 partes, testa com 1
3. Calcula a métrica R² em cada rodada
4. Retorna um array com 5 valores (um por rodada)

**O sinal negativo** em `neg_mean_absolute_error`: O scikit-learn padroniza que métricas "maiores = melhores". Como MAE é "menor = melhor", ele retorna o negativo. Por isso multiplicamos por -1.

---

## Célula 10 — Feature Importance

```python
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X, y)

importancias = pd.Series(rf.feature_importances_, index=features)
importancias = importancias.sort_values(ascending=True)
importancias.plot.barh()
```

### Explicando:

**`rf.fit(X, y)`** — Treina o modelo com TODOS os dados (não é validação cruzada aqui, é treino completo para analisar as features).

**`rf.feature_importances_`** — Depois de treinado, o modelo calcula automaticamente a importância de cada feature. Retorna um array com 9 valores (um por feature).

**`pd.Series(..., index=features)`** — Cria uma série nomeada (associa cada importância ao nome da feature).

**`.sort_values(ascending=True)`** — Ordena do menor para o maior (o mais importante fica embaixo no gráfico horizontal).

**`.plot.barh()`** — Gráfico de barras horizontal.

---

## Célula 11 — Prever e Analisar Resíduos

```python
melhor_modelo = modelos[melhor_nome]
melhor_modelo.fit(X, y)
y_pred = melhor_modelo.predict(X)

residuos = y - y_pred
```

**`y_pred = melhor_modelo.predict(X)`** — Usa o modelo treinado para prever a taxa de corrosão de cada anomalia.

**`residuos = y - y_pred`** — Calcula o erro de cada previsão:
- Se residuo = 0 → previsão perfeita
- Se residuo > 0 → modelo subestimou (previu menos que o real)
- Se residuo < 0 → modelo superestimou (previu mais que o real)

---

## Célula 12 — Simulação Temporal de uma Anomalia

```python
for i in range(len(anos_futuros) - 1):
    entrada = pd.DataFrame([{...}])        # Dados atuais da anomalia
    taxa_prevista = melhor_modelo.predict(entrada)[0]  # Modelo prevê a taxa
    prof_atual = min(prof_atual + taxa_prevista * 5, 100)  # Atualiza profundidade
```

### O que acontece:

1. Pega uma anomalia de exemplo (tipo CORR)
2. Para cada período de 5 anos (0→5, 5→10, 10→15, 15→20, 20→25, 25→30):
   - Monta uma "ficha" com os dados atuais da anomalia
   - Pede ao modelo: "qual a taxa de corrosão prevista?"
   - Calcula a nova profundidade: `profundidade + taxa × 5 anos`
   - Usa essa nova profundidade como entrada do próximo período
3. Plota um gráfico mostrando a evolução ao longo de 30 anos

É como um **efeito dominó**: a previsão de um período alimenta o próximo. Isso permite projetar a vida útil da anomalia.

---

## Resumo: As 3 Linhas Mais Importantes do Projeto

Se tivesse que reduzir todo o notebook a 3 linhas, seria:

```python
# 1. Criar o modelo
modelo = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)

# 2. Treinar com os dados
modelo.fit(X, y)

# 3. Prever dados novos
previsao = modelo.predict(dados_novos)
```

Todo o resto (limpeza, EDA, simulação, métricas, gráficos) é para **preparar os dados** antes dessas 3 linhas e **avaliar a qualidade** depois.
