# Aprendendo Machine Learning — Parte 1: Fundamentos

> Este documento explica os conceitos fundamentais de Machine Learning de forma simples e direta, usando o nosso projeto de corrosão como exemplo.

---

## 1. O que é Machine Learning?

Machine Learning (Aprendizado de Máquina) é uma forma de ensinar o computador a **aprender padrões a partir de dados**, sem que você precise programar cada regra manualmente.

### Analogia simples

Imagine que você é um engenheiro experiente. Depois de analisar centenas de anomalias em dutos, você começa a "sentir" que:
- Anomalias mais profundas tendem a crescer mais rápido
- Corrosão ativa (CORR) é mais agressiva que defeitos de fabricação
- Tubos com parede mais fina sofrem mais

Você aprendeu essas regras pela **experiência** (dados). Machine Learning faz a mesma coisa, mas com matemática e computação — e pode encontrar padrões que um humano talvez não perceberia.

---

## 2. Tipos de Machine Learning

Existem 3 grandes categorias:

### 2.1 Aprendizado Supervisionado (o que usamos)

Você dá ao modelo:
- **Dados de entrada** (features/características): espessura, tipo, profundidade...
- **Resposta correta** (target/alvo): a taxa de corrosão que realmente aconteceu

O modelo aprende a relação entre entrada e saída. Depois, quando receber dados novos (sem resposta), ele prevê.

```
TREINAMENTO:
  Entrada: [espessura=7.1, tipo=CORR, prof=15%]  →  Resposta: 0.8 %/ano
  Entrada: [espessura=8.7, tipo=ASCI, prof=10%]  →  Resposta: 0.3 %/ano
  ... (centenas de exemplos)

PREVISÃO:
  Entrada: [espessura=7.1, tipo=CORR, prof=20%]  →  Modelo prevê: ? %/ano
```

### 2.2 Aprendizado Não-Supervisionado

Não tem resposta correta. O modelo tenta encontrar grupos ou padrões sozinho. Ex: agrupar anomalias similares automaticamente. **Não usamos neste projeto.**

### 2.3 Aprendizado por Reforço

O modelo aprende por tentativa e erro, recebendo recompensas. Ex: treinar um robô. **Não usamos neste projeto.**

---

## 3. Regressão vs Classificação

Dentro do aprendizado supervisionado, existem dois tipos de problema:

| Tipo           | O que prevê                    | Exemplo no nosso contexto                    |
|----------------|--------------------------------|----------------------------------------------|
| **Regressão**  | Um **número contínuo**         | "A taxa de corrosão será 0.65 %/ano"         |
| Classificação  | Uma **categoria**              | "Esta anomalia é crítica ou não-crítica"      |

**Nosso projeto é de REGRESSÃO** — queremos prever um número (a taxa de corrosão em %/ano).

---

## 4. Vocabulário Essencial

Estes são os termos que você vai encontrar no código e na documentação:

### 4.1 Dataset (Conjunto de Dados)

A tabela com todos os dados. Cada **linha** é um exemplo (uma anomalia), cada **coluna** é uma informação.

```
| espessura | tipo | profundidade | taxa_corrosao |
|-----------|------|-------------|---------------|
|    7.1    | CORR |     15      |     0.82      |  ← Um exemplo
|    8.7    | ASCI |     10      |     0.35      |  ← Outro exemplo
|    7.1    | COSC |     12      |     0.28      |
```

### 4.2 Features (Características / Variáveis de Entrada)

São as **colunas que o modelo usa para aprender**. No nosso caso:
- espessura_mm
- comprimento_mm
- largura_mm
- profundidade_inicial_pct
- erf
- tipo (codificado como número)
- tipo_pof (codificado como número)
- i_e (codificado como número)
- ano_inicio

Chamamos de **X** no código.

### 4.3 Target (Alvo / Variável de Saída)

É a **coluna que queremos prever**. No nosso caso:
- `taxa_corrosao_pct_ano` — a taxa de corrosão em %/ano

Chamamos de **y** no código.

### 4.4 Treino e Teste

Para saber se o modelo realmente aprendeu (e não apenas "decorou"), dividimos os dados:

```
Dataset total (846 exemplos)
    ├── Dados de Treino (~80%) → O modelo aprende com estes
    └── Dados de Teste (~20%)  → Testamos se o modelo acerta estes
                                  (ele NUNCA viu estes dados antes)
```

Se o modelo acerta bem nos dados de teste, significa que ele **generalizou** — aprendeu o padrão real, não decorou.

### 4.5 Validação Cruzada (Cross-Validation)

É uma versão mais robusta da divisão treino/teste. Em vez de dividir uma vez só, dividimos **5 vezes** de formas diferentes:

```
Rodada 1: [TESTE][Treino][Treino][Treino][Treino]
Rodada 2: [Treino][TESTE][Treino][Treino][Treino]
Rodada 3: [Treino][Treino][TESTE][Treino][Treino]
Rodada 4: [Treino][Treino][Treino][TESTE][Treino]
Rodada 5: [Treino][Treino][Treino][Treino][TESTE]
```

Cada rodada testa com uma parte diferente. O resultado final é a **média** das 5 rodadas. Isso é o **K-Fold** com K=5, que usamos no projeto.

### 4.6 Overfitting (Sobreajuste)

Quando o modelo "decora" os dados de treino mas erra nos dados novos. É como um aluno que decora as respostas da prova anterior mas não entende a matéria.

**Sinais de overfitting:**
- R² muito alto no treino (ex: 0.99) mas baixo no teste (ex: 0.50)
- O modelo é complexo demais para a quantidade de dados

### 4.7 Underfitting (Subajuste)

Quando o modelo é simples demais e não consegue capturar os padrões. É como usar uma régua para mapear uma curva.

**Sinais de underfitting:**
- R² baixo tanto no treino quanto no teste
- O modelo não melhora mesmo com mais dados

---

## 5. O Fluxo Completo de um Projeto de ML

```
┌─────────────────────────────────────────────────────────┐
│  1. COLETAR DADOS                                       │
│     Obter o arquivo Excel com dados da inspeção PIG     │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│  2. LIMPAR E PREPARAR                                   │
│     Converter tipos, tratar nulos, padronizar           │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│  3. EXPLORAR (EDA)                                      │
│     Gráficos, estatísticas, entender os dados           │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│  4. PREPARAR FEATURES                                   │
│     Selecionar colunas, codificar categorias            │
│     Definir X (features) e y (target)                   │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│  5. TREINAR MODELOS                                     │
│     Dar os dados de treino para o modelo aprender       │
│     modelo.fit(X_treino, y_treino)                      │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│  6. AVALIAR                                             │
│     Testar com dados que o modelo não viu               │
│     Calcular métricas (R², MAE, RMSE)                   │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│  7. USAR (PREVER)                                       │
│     Dar dados novos e obter previsões                   │
│     modelo.predict(dados_novos)                         │
└─────────────────────────────────────────────────────────┘
```

Esse é exatamente o fluxo que seguimos no notebook `estudo_corrosao.ipynb`.

---

## 6. Por que Precisamos Codificar Categorias?

Modelos de ML trabalham com **números**. Quando temos colunas de texto como "CORR", "ASCI", "COSC", precisamos convertê-las em números.

### LabelEncoder (o que usamos)

Atribui um número inteiro para cada categoria:

```
ASCI → 0
CORR → 1
COSC → 2
```

No código:
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['tipo_cod'] = le.fit_transform(df['tipo'])
# fit_transform = "aprenda as categorias" + "converta para números"
```

---

## 7. Resumo

| Conceito | No nosso projeto |
|----------|-----------------|
| Tipo de ML | Aprendizado Supervisionado |
| Tipo de problema | Regressão (prever um número) |
| Features (X) | 9 variáveis: espessura, tipo, profundidade, etc. |
| Target (y) | taxa_corrosao_pct_ano |
| Validação | Cross-Validation K-Fold (K=5) |
| Modelos testados | Regressão Linear, Random Forest, XGBoost |
| Melhor modelo | Random Forest (R² = 0.85) |
