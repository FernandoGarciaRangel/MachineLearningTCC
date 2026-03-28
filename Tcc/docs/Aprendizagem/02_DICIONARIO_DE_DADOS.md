# Dicionário de Dados

Este documento detalha todos os datasets utilizados no projeto, seus campos, tipos e transformações aplicadas.

---

## 1. Dataset Original — `exemplo passagem de pig.xlsx`

Dados brutos de uma inspeção por PIG em um duto de transporte.

| # | Coluna Original          | Nome Padronizado         | Tipo Original | Tipo Final | Unidade | Faixa Observada        | Descrição                                                                |
|---|--------------------------|--------------------------|---------------|------------|---------|------------------------|--------------------------------------------------------------------------|
| 1 | ID Tubo                  | `id_tubo`                | String        | String     | —       | ex: 191041102          | Identificador único do tubo (joint) no cadastro do duto                  |
| 2 | Posição (m)              | `posicao_m`              | String (BR)   | Float      | metros  | 18.017 — 62,525.259    | Distância da anomalia em relação ao ponto zero do duto                   |
| 3 | Dist. Sld. Ant. (m)     | `dist_sold_ant_m`        | String (BR)   | Float      | metros  | 0.000 — 12.451         | Distância até a solda circunferencial anterior mais próxima              |
| 4 | Compr. Tubo (m)          | `compr_tubo_m`           | String (BR)   | Float      | metros  | 5.348 — 12.453         | Comprimento total do tubo (joint) onde a anomalia está                   |
| 5 | I/E                      | `i_e`                    | String        | String     | —       | "-" (E), "I"           | Localização da anomalia: Interna (I) ou Externa (E, representada por "-")|
| 6 | Tipo                     | `tipo`                   | String        | String     | —       | ASCI, COSC, CORR       | Classificação do tipo de anomalia                                        |
| 7 | Posição Horária (hh:mm)  | `posicao_horaria`        | String        | String     | hh:mm   | 00:08 — 11:59          | Posição circunferencial no tubo, usando analogia de relógio (12h = topo) |
| 8 | Esp. (mm)                | `espessura_mm`           | String (BR)   | Float      | mm      | 7.1, 8.7               | Espessura nominal da parede do tubo naquele trecho                       |
| 9 | Compr. (mm)              | `comprimento_mm`         | String (BR)   | Float      | mm      | 10 — 89                | Dimensão axial (ao longo do eixo do duto) da anomalia                    |
| 10| Larg. (mm)               | `largura_mm`             | String (BR)   | Float      | mm      | 10 — 145               | Dimensão circunferencial da anomalia                                     |
| 11| Prof. (%)                | `profundidade_pct`       | String (BR)   | Float      | %       | 10 — 39                | Profundidade da anomalia como % da espessura da parede                   |
| 12| ERF                      | `erf`                    | String (BR)   | Float      | —       | 0.46 — 0.77            | Estimated Repair Factor: razão entre pressão de falha e pressão operação |
| 13| Tipo POF                 | `tipo_pof`               | String        | String     | —       | CIGR, GENE, PITT       | Classificação do modo de falha provável                                  |

> **Nota:** "String (BR)" indica que o valor original é uma string com formatação numérica brasileira (ex: "1.234,56").

---

## 2. Dataset para Machine Learning — `df_ml`

Gerado a partir do dataset original + simulação de crescimento. Cada registro representa o comportamento de uma anomalia em um intervalo de 5 anos.

| # | Campo                       | Tipo   | Unidade | Descrição                                                        |
|---|-----------------------------|--------|---------|------------------------------------------------------------------|
| 1 | `id_tubo`                   | String | —       | Identificador do tubo (herdado do original)                      |
| 2 | `tipo`                      | String | —       | Tipo de anomalia: ASCI, COSC, CORR                               |
| 3 | `tipo_pof`                  | String | —       | Tipo de probabilidade de falha: CIGR, GENE, PITT                 |
| 4 | `i_e`                       | String | —       | Interna (I) ou Externa (E)                                       |
| 5 | `espessura_mm`              | Float  | mm      | Espessura da parede do tubo                                      |
| 6 | `comprimento_mm`            | Float  | mm      | Comprimento axial da anomalia                                    |
| 7 | `largura_mm`                | Float  | mm      | Largura circunferencial da anomalia                              |
| 8 | `profundidade_inicial_pct`  | Float  | %       | Profundidade no início do intervalo                              |
| 9 | `profundidade_final_pct`    | Float  | %       | Profundidade ao final do intervalo (após crescimento)            |
| 10| `erf`                       | Float  | —       | Fator de reparo estimado                                         |
| 11| `ano_inicio`                | Int    | ano     | Ano de início do intervalo (0, 5 ou 10)                          |
| 12| `ano_fim`                   | Int    | ano     | Ano de fim do intervalo (5, 10 ou 15)                            |
| 13| `intervalo_anos`            | Int    | anos    | Duração do intervalo (sempre 5)                                  |
| 14| `taxa_corrosao_pct_ano`     | Float  | %/ano   | **TARGET** — Taxa de crescimento da profundidade por ano         |

### Campos Derivados (encoding para ML)

| Campo             | Origem     | Mapeamento               |
|-------------------|------------|--------------------------|
| `tipo_cod`        | `tipo`     | ASCI=0, CORR=1, COSC=2  |
| `tipo_pof_cod`    | `tipo_pof` | CIGR=0, GENE=1, PITT=2  |
| `i_e_cod`         | `i_e`      | E=0, I=1                |

---

## 3. Transformações Aplicadas

### 3.1 Limpeza Numérica
```
Entrada:  "1.234,56"  (formato brasileiro)
Processo: remove "." → "1234,56" → substitui "," por "." → "1234.56"
Saída:    1234.56     (float Python)
```

### 3.2 Tratamento do Campo I/E
```
Entrada:  "-"   →  Saída: "E" (Externa)
Entrada:  "I"   →  Saída: "I" (Interna, mantida)
```

### 3.3 Cálculo da Taxa de Corrosão (simulada)
```
taxa = taxa_base[tipo] × (1 + prof_atual/100) × (8.0 / espessura) × N(1.0, 0.15)
taxa = max(taxa, 0.05)
profundidade_final = min(prof_atual + taxa × intervalo, 95)
```

---

## 4. Estatísticas Descritivas do Dataset Original

| Variável            | Contagem | Média     | Desvio Padrão | Mín    | 25%    | 50%    | 75%    | Máx       |
|---------------------|----------|-----------|---------------|--------|--------|--------|--------|-----------|
| posicao_m           | 282      | ~26,000   | —             | 18.0   | —      | —      | —      | 62,525.3  |
| espessura_mm        | 282      | ~7.3      | —             | 7.1    | 7.1    | 7.1    | 7.1    | 8.7       |
| comprimento_mm      | 282      | ~17.0     | —             | 10.0   | 10.0   | 11.0   | 18.0   | 89.0      |
| largura_mm          | 282      | ~65.0     | —             | 10.0   | 41.0   | 60.0   | 88.0   | 145.0     |
| profundidade_pct    | 282      | ~14.5     | —             | 10.0   | 11.0   | 13.0   | 16.0   | 39.0      |
| erf                 | 282      | ~0.74     | —             | 0.46   | 0.76   | 0.76   | 0.76   | 0.77      |

### Distribuição por Tipo de Anomalia
| Tipo | Quantidade | Percentual |
|------|-----------|------------|
| ASCI | ~150      | ~53%       |
| COSC | ~80       | ~28%       |
| CORR | ~52       | ~18%       |

### Distribuição por Tipo POF
| Tipo POF | Quantidade |
|----------|-----------|
| CIGR     | Maioria   |
| GENE     | Segundo   |
| PITT     | Menor     |
