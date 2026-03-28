# Aprendendo Machine Learning — Parte 4: Métricas e Avaliação

> Este documento explica como saber se um modelo de ML é bom ou ruim, o que cada métrica significa e como interpretar os gráficos gerados.

---

## 1. Por que precisamos de métricas?

Treinar um modelo é fácil — 3 linhas de código. A parte difícil é saber: **ele é bom o suficiente?**

Métricas são "notas" que damos ao modelo. Assim como uma prova avalia um aluno, métricas avaliam se o modelo aprendeu de verdade.

---

## 2. As 3 Métricas que Usamos

### 2.1 R² (R-Quadrado / Coeficiente de Determinação)

**O que mede:** Qual porcentagem da variação nos dados o modelo consegue explicar.

**Faixa:** 0.0 a 1.0 (pode ser negativo se o modelo for péssimo)

**Analogia:** Imagine que a taxa de corrosão varia de 0.1 a 1.5 %/ano. O R² diz quanto dessa variação o modelo captura:

```
R² = 0.00 → O modelo é tão ruim quanto chutar a média sempre
R² = 0.50 → O modelo explica 50% da variação (mediano)
R² = 0.85 → O modelo explica 85% da variação (bom!)
R² = 1.00 → Previsão perfeita (impossível na prática)
```

**No nosso projeto:**
| Modelo | R² | Interpretação |
|--------|-----|--------------|
| Regressão Linear | 0.76 | Explica 76% — razoável |
| Random Forest | **0.85** | Explica 85% — bom |
| XGBoost | 0.83 | Explica 83% — bom |

**Fórmula (simplificada):**
```
R² = 1 - (soma dos erros do modelo² / soma da variação total²)
```

Se os erros do modelo são muito menores que a variação natural, R² fica próximo de 1.

---

### 2.2 MAE (Mean Absolute Error / Erro Absoluto Médio)

**O que mede:** Na média, quanto o modelo erra em cada previsão (em valor absoluto).

**Unidade:** A mesma do target — no nosso caso, **%/ano**

**Analogia:** Se o MAE = 0.08, significa que, em média, a previsão do modelo erra por 0.08 %/ano para mais ou para menos.

```
Exemplo:
  Taxa real:    0.65 %/ano
  Previsto:     0.72 %/ano
  Erro:         |0.65 - 0.72| = 0.07

  Taxa real:    0.40 %/ano
  Previsto:     0.31 %/ano
  Erro:         |0.40 - 0.31| = 0.09

  MAE = média(0.07, 0.09) = 0.08 %/ano
```

**No nosso projeto:**
| Modelo | MAE | Interpretação |
|--------|-----|---------------|
| Regressão Linear | 0.094 | Erra ~0.09 %/ano em média |
| Random Forest | **0.078** | Erra ~0.08 %/ano em média |
| XGBoost | 0.084 | Erra ~0.08 %/ano em média |

**Quanto é "bom"?** Depende do contexto. Nossa taxa média é ~0.61 %/ano. Um MAE de 0.08 representa um erro de ~13% em relação à média — aceitável para uma prova de conceito.

---

### 2.3 RMSE (Root Mean Squared Error / Raiz do Erro Quadrático Médio)

**O que mede:** Similar ao MAE, mas **penaliza mais os erros grandes**.

**Unidade:** Mesma do target — **%/ano**

**Diferença entre MAE e RMSE:**

```
Previsões:   Erro = [0.05, 0.05, 0.05, 0.50]

MAE  = média(0.05, 0.05, 0.05, 0.50)
     = 0.1625

RMSE = raiz(média(0.05², 0.05², 0.05², 0.50²))
     = raiz(média(0.0025, 0.0025, 0.0025, 0.25))
     = raiz(0.0644)
     = 0.2537   ← Maior! O erro grande (0.50) foi penalizado
```

**Quando RMSE >> MAE:** Indica que existem alguns erros muito grandes (outliers nas previsões).

**Quando RMSE ≈ MAE:** Indica que os erros são uniformes (todos parecidos).

**No nosso projeto:**
| Modelo | MAE | RMSE | RMSE/MAE |
|--------|-----|------|----------|
| Regressão Linear | 0.094 | 0.137 | 1.46 |
| Random Forest | 0.078 | 0.108 | 1.38 |
| XGBoost | 0.084 | 0.115 | 1.37 |

A razão RMSE/MAE está entre 1.3-1.5 para todos — os erros são relativamente uniformes, sem outliers extremos.

---

## 3. Resumo das Métricas

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  R²    → "Quão bem o modelo explica os dados?"                 │
│           Quanto MAIOR, melhor (ideal = 1.0)                   │
│                                                                │
│  MAE   → "Quanto erra em média?"                               │
│           Quanto MENOR, melhor (ideal = 0.0)                   │
│                                                                │
│  RMSE  → "Quanto erra, penalizando erros grandes?"             │
│           Quanto MENOR, melhor (ideal = 0.0)                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. Interpretando os Gráficos do Notebook

### 4.1 Gráfico de Barras Comparativo (Seção 6)

```
  R²                    MAE                   RMSE
 ┌────┐               ┌────┐               ┌────┐
 │0.85│ ← melhor      │    │               │    │
 │    │  ┌────┐       │    │  ┌────┐       │    │  ┌────┐
 │    │  │0.83│       │    │  │0.08│       │    │  │0.12│
 │    │  │    │┌────┐ │0.09│  │    │┌────┐ │0.14│  │    │┌────┐
 │    │  │    ││0.76│ │    │  │    ││0.08│ │    │  │    ││0.11│
 │ RF │  │XGB ││ LR │ │ LR │  │XGB ││ RF │ │ LR │  │XGB ││ RF │
 └────┘  └────┘└────┘ └────┘  └────┘└────┘ └────┘  └────┘└────┘
  Maior = Melhor        Menor = Melhor        Menor = Melhor
```

**Como ler:** Para R², a barra mais alta é melhor. Para MAE e RMSE, a barra mais baixa é melhor. O Random Forest vence nas 3 métricas.

---

### 4.2 Gráfico Previsto vs Real (Seção 8)

```
Previsto
    │       •  •  •
    │     •  / •
    │   •  /•  •
    │  • /•
    │ •/•
    │/•
    └─────────────────
         Real
```

**Como ler:**
- A **linha vermelha tracejada** é a previsão perfeita (previsto = real)
- Cada **ponto azul** é uma anomalia
- Quanto mais **perto da linha** os pontos estiverem, melhor o modelo
- Pontos distantes da linha são erros grandes

**Bom modelo:** Pontos formando uma "faixa" estreita em torno da linha diagonal.

**Modelo ruim:** Pontos espalhados aleatoriamente, sem seguir a diagonal.

---

### 4.3 Distribuição dos Resíduos (Seção 8)

Resíduo = Valor Real - Valor Previsto

```
Frequência
    │
    │      ████
    │    ████████
    │  ████████████
    │████████████████
    └────────┬────────
            0.0
          Resíduo
```

**Como ler:**
- O pico deve estar em **0** (a maioria dos erros é zero ou quase zero)
- A distribuição deve ser **simétrica** (erra tanto para mais quanto para menos)
- Se o pico estiver deslocado do zero → o modelo tem viés (erra sistematicamente para um lado)
- Se tiver "caudas" longas → existem erros grandes ocasionais

---

### 4.4 Gráfico de Evolução Temporal (Seção 9)

```
Profundidade (%)
 100│
    │
  80│- - - - - - - - - - - - -  ← Limite crítico
    │                     •
  60│               •  /
    │          •  /
  40│     •  /
    │  • /
  20│ •
    │•
   0└───────────────────────────
    0    5   10   15   20   25  30
                  Anos
```

**Como ler:**
- **Linha vermelha com pontos:** Trajetória prevista da profundidade ao longo dos anos
- **Linha laranja tracejada (80%):** Limite típico de reparo — quando a profundidade atinge 80%, a anomalia é considerada crítica
- **O cruzamento das linhas** indica o ano estimado em que será necessária intervenção
- A **área vermelha transparente** é apenas visual, para dar destaque à região sob a curva

---

## 5. O que é Validação Cruzada (no detalhe)

A validação cruzada K-Fold que usamos funciona assim:

```
Dados: [A][B][C][D][E]   (5 partes iguais de ~170 registros cada)

Rodada 1: Treina com [B][C][D][E], testa com [A] → R² = 0.84
Rodada 2: Treina com [A][C][D][E], testa com [B] → R² = 0.86
Rodada 3: Treina com [A][B][D][E], testa com [C] → R² = 0.83
Rodada 4: Treina com [A][B][C][E], testa com [D] → R² = 0.87
Rodada 5: Treina com [A][B][C][D], testa com [E] → R² = 0.85

Resultado: R² = média(0.84, 0.86, 0.83, 0.87, 0.85) = 0.85
Desvio:    ±0.01 (todas as rodadas deram resultados similares = modelo estável)
```

**Por que fazemos isso em vez de um único treino/teste?**
- Um único split pode ser "sortudo" ou "azarado"
- Com 5 rodadas, temos 5 estimativas independentes
- O desvio padrão (±) mostra se o modelo é estável ou oscila muito

**No código:**
```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(modelo, X, y, cv=kf, scoring='r2')
# scores é um array com 5 valores, ex: [0.84, 0.86, 0.83, 0.87, 0.85]
# scores.mean() = 0.85
# scores.std()  = 0.01
```

---

## 6. Checklist: Como Saber se seu Modelo é Bom

- [ ] **R² > 0.70** em validação cruzada? (aceitável para dados reais de engenharia)
- [ ] **R² no teste ≈ R² no treino?** (se o treino for muito maior, é overfitting)
- [ ] **MAE faz sentido no contexto?** (erro de 0.08 %/ano é aceitável para taxa média de 0.61?)
- [ ] **RMSE / MAE < 1.5?** (se muito maior, existem outliers problemáticos)
- [ ] **Resíduos centrados em zero?** (sem viés sistemático)
- [ ] **Resíduos simétricos?** (erra igualmente para mais e para menos)
- [ ] **Feature importance faz sentido?** (as features mais importantes são lógicas no domínio?)
- [ ] **Desvio padrão do R² é baixo?** (modelo estável entre as rodadas)

No nosso projeto, o Random Forest passa em todos esses critérios.
