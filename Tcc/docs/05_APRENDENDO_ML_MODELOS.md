# Aprendendo Machine Learning — Parte 2: Os 3 Modelos Explicados

> Este documento explica como cada modelo usado no projeto funciona internamente, com analogias e exemplos visuais.

---

## 1. Regressão Linear

### Como funciona?

É o modelo mais simples. Ele tenta traçar uma **linha reta** (ou um plano, com múltiplas variáveis) que melhor se ajusta aos dados.

### Analogia

Imagine que você tem um gráfico com:
- Eixo X: profundidade da anomalia (%)
- Eixo Y: taxa de corrosão (%/ano)

A Regressão Linear tenta encontrar a melhor reta que passa no meio dos pontos:

```
Taxa (%/ano)
     │
 1.2 │                              •
 1.0 │                    •    •  /
 0.8 │              •   •    /
 0.6 │        •   •    / •
 0.4 │    •  •  /  •
 0.2 │  •  /•
 0.0 │/
     └───────────────────────────────
     10   15   20   25   30   35   40
                Profundidade (%)
```

A fórmula é:
```
taxa = a₁×espessura + a₂×profundidade + a₃×comprimento + ... + b
```

O modelo aprende os valores de a₁, a₂, a₃... (chamados **coeficientes**) e b (chamado **intercepto**) que minimizam o erro.

### No código

```python
from sklearn.linear_model import LinearRegression

modelo = LinearRegression()
modelo.fit(X, y)           # Treina: encontra os coeficientes
previsao = modelo.predict(X_novo)  # Prevê com dados novos
```

### Vantagens
- Simples e rápido
- Fácil de interpretar (cada coeficiente mostra o "peso" de cada feature)
- Bom como baseline (referência para comparar com modelos mais complexos)

### Limitações
- Só captura relações **lineares** (retas)
- Se a relação real for curva, ele erra bastante
- No nosso projeto: R² = 0.76 — razoável, mas os outros modelos foram melhores

### Quando a Regressão Linear falha

Se a relação entre profundidade e taxa de corrosão for assim:

```
Taxa
  │         •  •
  │       •      •
  │     •          •
  │   •
  │  •
  │•
  └────────────────────
     Profundidade
```

Uma reta não consegue capturar essa curva. Precisamos de modelos mais flexíveis.

---

## 2. Random Forest (Floresta Aleatória)

### Como funciona?

O Random Forest é uma coleção de **muitas árvores de decisão** que trabalham juntas. Cada árvore "vota" e o resultado final é a **média dos votos**.

### Passo 1: Entender uma Árvore de Decisão

Uma árvore faz perguntas sequenciais para chegar a uma resposta:

```
                    ┌─────────────────────┐
                    │ profundidade > 20%? │
                    └─────────┬───────────┘
                       sim/        \não
                      /              \
         ┌──────────────────┐   ┌──────────────────┐
         │ tipo == CORR?    │   │ espessura < 7.5? │
         └────────┬─────────┘   └────────┬─────────┘
            sim/     \não          sim/     \não
           /          \           /          \
    ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
    │taxa=0.95 │ │taxa=0.60 │ │taxa=0.40 │ │taxa=0.25 │
    └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

Para prever uma anomalia com profundidade=25%, tipo=CORR:
1. profundidade > 20%? **SIM** → vai para a esquerda
2. tipo == CORR? **SIM** → vai para a esquerda
3. Resultado: taxa = 0.95 %/ano

### Passo 2: De uma árvore para uma floresta

Uma árvore sozinha tende a **decorar** os dados (overfitting). A solução: criar **200 árvores** (no nosso caso), cada uma treinada com:
- Uma **amostra aleatória** dos dados (nem todas as árvores veem todos os dados)
- Um **subconjunto aleatório** das features (nem todas as árvores usam todas as colunas)

```
┌──────────┐  ┌──────────┐  ┌──────────┐       ┌──────────┐
│ Árvore 1 │  │ Árvore 2 │  │ Árvore 3 │  ...  │Árvore 200│
│ prev=0.82│  │ prev=0.91│  │ prev=0.78│       │ prev=0.85│
└──────────┘  └──────────┘  └──────────┘       └──────────┘
      \              \            /                  /
       \              \          /                  /
        ────────── MÉDIA ──────────
              prev = 0.84 %/ano
```

A média de 200 opiniões diferentes é muito mais confiável que uma só.

### No código

```python
from sklearn.ensemble import RandomForestRegressor

modelo = RandomForestRegressor(
    n_estimators=200,   # Número de árvores na floresta
    max_depth=10,       # Profundidade máxima de cada árvore
    random_state=42     # Seed para reprodutibilidade
)
modelo.fit(X, y)
previsao = modelo.predict(X_novo)
```

**Explicando os parâmetros:**

| Parâmetro | O que faz | No nosso caso |
|-----------|-----------|---------------|
| `n_estimators` | Quantas árvores criar | 200 (mais árvores = mais estável, mas mais lento) |
| `max_depth` | Quantos níveis de perguntas cada árvore pode ter | 10 (limitar evita overfitting) |
| `random_state` | Semente aleatória (garante mesmo resultado sempre) | 42 (número arbitrário) |

### Vantagens
- Captura relações **não-lineares** (curvas, interações entre variáveis)
- Muito robusto contra overfitting (graças à aleatoriedade)
- Funciona bem com **poucos dados** (nosso caso: 846 exemplos)
- Dá a **importância de cada feature** automaticamente
- Não precisa normalizar os dados

### Por que foi o melhor no nosso projeto?
- Os dados têm relações não-lineares (a taxa não cresce em linha reta com a profundidade)
- O dataset é relativamente pequeno (846 registros) — RF lida bem com isso
- R² = 0.85 — explica 85% da variação nos dados

---

## 3. XGBoost (Extreme Gradient Boosting)

### Como funciona?

O XGBoost também usa árvores de decisão, mas de forma diferente do Random Forest. Em vez de criar árvores **independentes** e fazer a média, ele cria árvores **sequenciais** onde cada nova árvore **corrige os erros** da anterior.

### Analogia

Imagine que você está desenhando um retrato:

```
Árvore 1: Faz um rascunho grosseiro
           (captura o padrão geral)
              ↓
Árvore 2: Olha para os ERROS do rascunho e corrige
           (ajusta onde errou mais)
              ↓
Árvore 3: Olha para os erros RESTANTES e corrige
           (refinamento fino)
              ↓
         ... (200 rodadas)
              ↓
Resultado: Um desenho muito detalhado
```

### Diferença entre Random Forest e XGBoost

```
RANDOM FOREST (árvores paralelas):
┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
│  A1 │ │  A2 │ │  A3 │ │ A200│   ← Todas independentes
└──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
   └───────┴───┬───┴───────┘
           MÉDIA
         Resultado

XGBOOST (árvores sequenciais):
┌─────┐     ┌─────┐     ┌─────┐         ┌─────┐
│  A1 │ ──→ │  A2 │ ──→ │  A3 │ ──→ ... │ A200│
└─────┘     └─────┘     └─────┘         └─────┘
 rascunho    corrige     refina          resultado
              erros       mais             final
```

### No código

```python
from xgboost import XGBRegressor

modelo = XGBRegressor(
    n_estimators=200,      # Número de rodadas de correção
    max_depth=6,           # Profundidade de cada árvore (menor que RF)
    learning_rate=0.1,     # Tamanho do passo de correção
    random_state=42
)
modelo.fit(X, y)
previsao = modelo.predict(X_novo)
```

**Explicando os parâmetros:**

| Parâmetro | O que faz | No nosso caso |
|-----------|-----------|---------------|
| `n_estimators` | Quantas rodadas de correção | 200 |
| `max_depth` | Complexidade de cada árvore | 6 (menor que RF porque cada árvore é só um ajuste) |
| `learning_rate` | O quanto cada árvore corrige (0.0 a 1.0) | 0.1 (correções pequenas = mais estável) |
| `random_state` | Semente aleatória | 42 |

**O learning_rate é crucial:**
- **Alto (ex: 1.0):** Cada árvore corrige muito → aprende rápido mas pode oscilar
- **Baixo (ex: 0.01):** Cada árvore corrige pouco → mais estável mas precisa de mais rodadas
- **0.1:** Um bom equilíbrio (padrão da indústria)

### Vantagens
- Geralmente o **melhor modelo para dados tabulares** em competições (Kaggle)
- Muito eficiente em termos de tempo
- Tem regularização embutida (evita overfitting)
- Lida com valores faltantes automaticamente

### Por que ficou em segundo no nosso projeto?
- Com datasets pequenos, o XGBoost às vezes não tem dados suficientes para o processo de correção sequencial brilhar
- Random Forest é naturalmente mais robusto com poucos dados
- A diferença foi pequena: RF (0.85) vs XGB (0.83) — em dados reais pode inverter

---

## 4. Comparação Visual dos 3 Modelos

```
                    Regressão Linear    Random Forest     XGBoost
                    ────────────────    ─────────────     ───────
Complexidade:       Baixa               Média             Média-Alta
Como aprende:       Uma fórmula reta    200 árvores       200 árvores
                                        independentes     sequenciais
Captura curvas:     Não                 Sim               Sim
Risco overfitting:  Baixo               Baixo             Médio
Velocidade treino:  Muito rápida        Média             Rápida
Interpretabilidade: Alta                Média             Baixa
Precisa normalizar: Sim (recomendado)   Não               Não

Resultado no projeto:
  R²  =              0.76               0.85              0.83
  MAE =              0.094              0.078             0.084
  RMSE =             0.137              0.108             0.115
```

---

## 5. O que é .fit() e .predict()?

Esses são os dois métodos fundamentais de qualquer modelo no scikit-learn:

### .fit(X, y) — Treinar

"Modelo, aqui estão os dados (X) e as respostas corretas (y). Aprenda!"

```python
modelo.fit(X, y)
```

O que acontece por dentro:
- **Regressão Linear:** calcula os coeficientes da equação
- **Random Forest:** constrói 200 árvores de decisão
- **XGBoost:** constrói 200 árvores sequenciais, cada uma corrigindo a anterior

### .predict(X_novo) — Prever

"Modelo, aqui estão dados novos. O que você prevê?"

```python
previsao = modelo.predict(X_novo)
```

O que acontece por dentro:
- **Regressão Linear:** aplica a fórmula y = a₁x₁ + a₂x₂ + ... + b
- **Random Forest:** cada árvore vota, retorna a média
- **XGBoost:** soma as correções de todas as árvores

---

## 6. O que é Feature Importance?

Modelos baseados em árvores (RF e XGBoost) calculam automaticamente quais features foram **mais úteis** para fazer as previsões.

```
profundidade_inicial_pct  ████████████████████████  0.45  ← Mais importante!
espessura_mm              ██████████████            0.25
tipo_cod                  ████████                  0.15
erf                       ████                      0.08
comprimento_mm            ██                        0.03
largura_mm                █                         0.02
tipo_pof_cod              █                         0.01
i_e_cod                   ▏                         0.005
ano_inicio                ▏                         0.005
```

**Como interpretar:** A profundidade inicial é responsável por 45% do poder de previsão. Faz sentido — anomalias mais profundas têm mais material exposto e tendem a crescer mais rápido.

No código:
```python
modelo.fit(X, y)
importancias = modelo.feature_importances_  # Array com a importância de cada feature
```

---

## 7. Quando usar cada modelo?

| Situação | Modelo recomendado |
|----------|-------------------|
| Poucos dados (< 1000 registros) | **Random Forest** |
| Muitos dados (> 10,000) | **XGBoost** |
| Precisa explicar para não-técnicos | **Regressão Linear** |
| Baseline rápido para comparação | **Regressão Linear** |
| Competição/produção com dados tabulares | **XGBoost** |
| Primeira tentativa em qualquer problema | **Random Forest** (robusto e simples) |

No nosso projeto (282 anomalias × 3 intervalos = 846 registros), o **Random Forest** foi a escolha ideal.
