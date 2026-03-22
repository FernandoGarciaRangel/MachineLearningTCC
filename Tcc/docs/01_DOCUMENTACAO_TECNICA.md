# Documentação Técnica — Previsão da Taxa de Corrosão em Dutos por Machine Learning

## 1. Visão Geral do Projeto

### 1.1 Objetivo
Desenvolver um modelo de Machine Learning capaz de **prever a taxa de crescimento de corrosão** em anomalias detectadas em dutos de transporte (oleodutos/gasodutos), utilizando dados de inspeção por PIG (Pipeline Inspection Gauge).

### 1.2 Contexto
Dutos de transporte estão sujeitos a diversos mecanismos de degradação ao longo do tempo, como corrosão interna e externa, trincas em soldas e defeitos de fabricação. A inspeção por PIG é o principal método para identificar e dimensionar essas anomalias sem interromper a operação.

Com dados de múltiplas inspeções (realizadas tipicamente a cada 5 anos), é possível calcular a taxa de crescimento de cada anomalia e, com isso, treinar modelos preditivos que auxiliem na **gestão da integridade estrutural** do duto.

### 1.3 Escopo Atual
Este estudo é uma **prova de conceito (PoC)** baseada em uma única inspeção real. Para suprir a ausência de dados temporais reais, foi implementada uma **simulação de crescimento de corrosão** com parâmetros realistas, permitindo demonstrar todo o pipeline de ML.

---

## 2. Dados de Entrada

### 2.1 Fonte
Arquivo: `exemplo passagem de pig.xlsx`
- **Origem:** Relatório de inspeção por PIG
- **Formato:** Excel (.xlsx), 1 aba, 283 linhas (1 cabeçalho + 282 anomalias)
- **Colunas:** 13

### 2.2 Descrição das Colunas Originais

| Coluna Original       | Nome Padronizado       | Tipo     | Descrição                                                              |
|-----------------------|------------------------|----------|------------------------------------------------------------------------|
| ID Tubo               | `id_tubo`              | String   | Identificador único do tubo onde a anomalia foi detectada              |
| Posição (m)           | `posicao_m`            | Float    | Posição da anomalia ao longo do duto, em metros                       |
| Dist. Sld. Ant. (m)   | `dist_sold_ant_m`      | Float    | Distância até a solda anterior mais próxima, em metros                |
| Compr. Tubo (m)       | `compr_tubo_m`         | Float    | Comprimento do tubo onde a anomalia se encontra, em metros            |
| I/E                   | `i_e`                  | String   | Indica se a anomalia é Interna (I) ou Externa (E). "-" = Externa     |
| Tipo                  | `tipo`                 | String   | Tipo de anomalia: ASCI, COSC ou CORR                                  |
| Posição Horária       | `posicao_horaria`      | String   | Posição circunferencial da anomalia no tubo (formato relógio hh:mm)   |
| Esp. (mm)             | `espessura_mm`         | Float    | Espessura nominal da parede do tubo, em milímetros                    |
| Compr. (mm)           | `comprimento_mm`       | Float    | Comprimento axial da anomalia, em milímetros                          |
| Larg. (mm)            | `largura_mm`           | Float    | Largura circunferencial da anomalia, em milímetros                    |
| Prof. (%)             | `profundidade_pct`     | Float    | Profundidade da anomalia como percentual da espessura da parede       |
| ERF                   | `erf`                  | Float    | Estimated Repair Factor — fator de resistência residual               |
| Tipo POF              | `tipo_pof`             | String   | Tipo de probabilidade de falha: CIGR, GENE ou PITT                   |

### 2.3 Tipos de Anomalia

| Código | Significado                    | Descrição                                              |
|--------|--------------------------------|--------------------------------------------------------|
| ASCI   | Associada à Costura Interna    | Defeito próximo ou sobre a solda longitudinal interna  |
| COSC   | Costura                        | Defeito na própria costura de solda                    |
| CORR   | Corrosão                       | Perda de metal por processo corrosivo                  |

### 2.4 Tipos POF (Probability of Failure)

| Código | Significado                    |
|--------|--------------------------------|
| CIGR   | Circunferencial Geral          |
| GENE   | Generalizada                   |
| PITT   | Pitting (corrosão localizada)  |

---

## 3. Pipeline de Processamento

### 3.1 Etapa 1 — Carregamento e Limpeza

**Problema:** Os dados do Excel estavam em formato texto com numeração brasileira (ponto como separador de milhar, vírgula como decimal).

**Solução implementada:**
```python
def limpar_numero(valor):
    """Converte string com formato brasileiro (1.234,56) para float."""
    if isinstance(valor, str):
        return float(valor.replace('.', '').replace(',', '.'))
    return float(valor)
```

**Transformações realizadas:**
- Conversão de 8 colunas de string para float
- Campo `i_e`: valor "-" substituído por "E" (Externa)
- Remoção da primeira linha (cabeçalho embutido nos dados)
- Reset do índice do DataFrame
- Verificação de valores nulos: **0 encontrados**

### 3.2 Etapa 2 — Análise Exploratória (EDA)

Foram geradas as seguintes análises visuais:

1. **Histogramas com KDE** das 6 variáveis numéricas principais (profundidade, comprimento, largura, espessura, ERF, posição)
2. **Gráficos de barras** para variáveis categóricas (tipo de anomalia, tipo POF, interna/externa)
3. **Mapa de calor de correlação** entre variáveis numéricas
4. **Estatísticas descritivas** (describe) de todas as variáveis numéricas

### 3.3 Etapa 3 — Simulação de Inspeções Futuras

Como o projeto dispõe de apenas uma inspeção, foi necessário **simular o crescimento temporal** das anomalias para gerar a variável alvo (taxa de corrosão).

**Parâmetros da simulação:**

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| Intervalo entre inspeções | 5 anos | Padrão da indústria |
| Número de inspeções simuladas | 4 (anos 0, 5, 10, 15) | 3 intervalos de crescimento por anomalia |
| Seed aleatória | 42 | Reprodutibilidade |

**Modelo de crescimento:**

A taxa de corrosão de cada anomalia é calculada como:

```
taxa = taxa_base × fator_profundidade × fator_espessura × ruído
```

Onde:
- **taxa_base**: depende do tipo de anomalia
  - CORR: 0.8 %/ano (corrosão ativa, maior taxa)
  - ASCI: 0.4 %/ano
  - COSC: 0.3 %/ano
- **fator_profundidade**: `1 + (profundidade_atual / 100)` — anomalias mais profundas crescem mais rápido
- **fator_espessura**: `8.0 / espessura_mm` — paredes mais finas são mais impactadas
- **ruído**: `N(1.0, 0.15)` — distribuição normal com média 1 e desvio 0.15
- **taxa mínima**: 0.05 %/ano (floor para evitar taxas negativas)
- **profundidade máxima**: 95% (cap para evitar valores irreais)

**Resultado:** 846 registros (282 anomalias × 3 intervalos)

### 3.4 Etapa 4 — Engenharia de Features

**Variáveis categóricas** foram codificadas com `LabelEncoder`:
- `tipo` → `tipo_cod` (ASCI=0, CORR=1, COSC=2)
- `tipo_pof` → `tipo_pof_cod` (CIGR=0, GENE=1, PITT=2)
- `i_e` → `i_e_cod` (E=0, I=1)

**Features utilizadas (9):**

| Feature                    | Tipo    | Descrição                                 |
|----------------------------|---------|-------------------------------------------|
| `espessura_mm`             | Float   | Espessura da parede do tubo               |
| `comprimento_mm`           | Float   | Comprimento axial da anomalia             |
| `largura_mm`               | Float   | Largura circunferencial da anomalia       |
| `profundidade_inicial_pct` | Float   | Profundidade no início do intervalo       |
| `erf`                      | Float   | Fator de reparo estimado                  |
| `tipo_cod`                 | Int     | Tipo de anomalia (codificado)             |
| `tipo_pof_cod`             | Int     | Tipo de probabilidade de falha (codificado)|
| `i_e_cod`                  | Int     | Interna/Externa (codificado)              |
| `ano_inicio`               | Int     | Ano de início do intervalo                |

**Variável alvo (target):**
- `taxa_corrosao_pct_ano`: taxa de crescimento da profundidade em %/ano

---

## 4. Modelos de Machine Learning

### 4.1 Modelos Avaliados

| Modelo               | Biblioteca     | Hiperparâmetros Principais                                      |
|----------------------|----------------|-----------------------------------------------------------------|
| Regressão Linear     | scikit-learn   | Padrão (sem regularização)                                      |
| Random Forest        | scikit-learn   | `n_estimators=200`, `max_depth=10`, `random_state=42`           |
| XGBoost              | xgboost        | `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `random_state=42` |

### 4.2 Validação

- **Método:** Validação cruzada K-Fold (k=5)
- **Métricas:**
  - **R² (Coeficiente de Determinação):** Mede o quanto o modelo explica da variância dos dados. Quanto mais próximo de 1, melhor.
  - **MAE (Mean Absolute Error):** Erro médio absoluto em %/ano. Quanto menor, melhor.
  - **RMSE (Root Mean Squared Error):** Raiz do erro quadrático médio. Penaliza mais erros grandes.

### 4.3 Resultados

| Modelo             | R²     | MAE    | RMSE   |
|--------------------|--------|--------|--------|
| Regressão Linear   | 0.7571 | 0.0938 | 0.1371 |
| **Random Forest**  | **0.8496** | **0.0776** | **0.1075** |
| XGBoost            | 0.8280 | 0.0844 | 0.1152 |

**Melhor modelo: Random Forest Regressor**

### 4.4 Importância das Features

Os modelos baseados em árvores permitem analisar quais variáveis mais contribuem para a previsão. As features mais importantes identificadas foram:

1. **profundidade_inicial_pct** — A profundidade atual é o principal preditor da taxa futura
2. **espessura_mm** — Espessura do tubo influencia diretamente a progressão
3. **tipo_cod** — O mecanismo de degradação determina a velocidade de crescimento
4. **erf** — O fator de reparo está correlacionado com a severidade

### 4.5 Análise de Resíduos

- Gráfico de **Previsto vs Real**: pontos concentrados ao longo da diagonal, indicando boa aderência
- **Distribuição dos resíduos**: centrada em zero, aproximadamente simétrica

---

## 5. Simulação Temporal

O notebook inclui uma demonstração de como o modelo pode projetar a **evolução da profundidade** de uma anomalia ao longo de 30 anos:

- Uma anomalia do tipo CORR é selecionada como exemplo
- O modelo prevê a taxa de corrosão para cada período de 5 anos
- A profundidade é atualizada incrementalmente
- O gráfico mostra a trajetória e um **limite crítico de 80%** como referência

Essa funcionalidade permite estimar **quando cada anomalia atingirá limites de reparo**, auxiliando no planejamento de manutenção.

---

## 6. Limitações e Ressalvas

1. **Dados simulados:** As taxas de corrosão foram geradas artificialmente. Os resultados dos modelos refletem a qualidade da simulação, não dados reais.
2. **Uma única inspeção:** Sem dados temporais reais, não é possível validar a acurácia das previsões.
3. **Pareamento de anomalias:** Em cenário real, o matching entre inspeções (mesma anomalia em datas diferentes) é um desafio significativo.
4. **Hiperparâmetros fixos:** Não foi realizada otimização de hiperparâmetros (GridSearch/RandomSearch).
5. **Sem tratamento de outliers:** Anomalias com valores extremos não foram filtradas.

---

## 7. Próximos Passos

1. **Integrar dados reais** de múltiplas inspeções (pelo menos 2 passagens de PIG)
2. **Implementar o pareamento** de anomalias entre inspeções por ID do tubo + posição
3. **Calcular taxas reais** de crescimento a partir das diferenças de profundidade
4. **Otimizar hiperparâmetros** com GridSearch ou Optuna
5. **Avaliar modelos probabilísticos** (Bayesian Ridge, Gaussian Process) para estimar incerteza
6. **Adicionar features derivadas:** taxa de crescimento anterior, idade do duto, condições operacionais
7. **Validar com normas técnicas:** comparar previsões com modelos normativos (ASME B31G, DNV-RP-F101)

---

## 8. Stack Tecnológica

| Componente       | Tecnologia          | Versão   |
|------------------|---------------------|----------|
| Linguagem        | Python              | 3.13.9   |
| Ambiente         | Anaconda (base)     | —        |
| IDE              | Cursor              | —        |
| Manipulação      | pandas              | 3.0.1+   |
| Numérico         | numpy               | 2.4.3+   |
| Visualização     | matplotlib          | 3.10.8   |
| Visualização     | seaborn             | 0.13.2   |
| ML               | scikit-learn        | 1.8.0    |
| ML (Boosting)    | xgboost             | 3.2.0    |
| Leitura Excel    | openpyxl            | 3.1.5    |
