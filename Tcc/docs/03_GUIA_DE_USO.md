# Guia de Uso e Reprodução

Este documento explica como configurar o ambiente, executar o notebook e adaptar o estudo para dados reais.

---

## 1. Pré-Requisitos

- **Python 3.10+** (testado com Anaconda 3.13.9 e Python 3.14)
- **pip** para instalação de pacotes
- **Cursor** ou **VS Code** com extensão Jupyter, ou **Jupyter Notebook/Lab**

---

## 2. Instalação do Ambiente

### 2.1 Instalar dependências

No terminal, na raiz do projeto (`MachineLearningTCC/`):

```bash
pip install -r requirements.txt
```

Conteúdo do `requirements.txt`:
```
pandas==3.0.1
numpy==2.4.3
scikit-learn==1.8.0
matplotlib==3.10.8
seaborn==0.13.2
xgboost==3.2.0
openpyxl==3.1.5
```

### 2.2 Configurar o kernel Jupyter (se necessário)

Se o notebook não encontrar as bibliotecas, pode ser que o kernel esteja apontando para outro Python. Para registrar o kernel correto:

```bash
pip install ipykernel
python -m ipykernel install --user --name python3 --display-name "Python 3"
```

Depois, selecione esse kernel no notebook (canto superior direito no Cursor/VS Code).

---

## 3. Estrutura do Projeto

```
MachineLearningTCC/
├── requirements.txt                  # Dependências do projeto
├── frutas.ipynb                      # Notebook de estudo (classificação de frutas)
├── Material de estudo/
│   └── frutas.ipynb                  # Material de referência
└── Tcc/
    ├── exemplo passagem de pig.xlsx  # Dados de inspeção PIG (entrada)
    ├── estudo_corrosao.ipynb         # Notebook principal do estudo
    └── docs/
        ├── 01_DOCUMENTACAO_TECNICA.md  # Documentação técnica completa
        ├── 02_DICIONARIO_DE_DADOS.md   # Dicionário de dados
        └── 03_GUIA_DE_USO.md           # Este documento
```

---

## 4. Executando o Notebook

### 4.1 Passos

1. Abra `Tcc/estudo_corrosao.ipynb` no Cursor ou VS Code
2. Selecione o kernel Python correto (com as libs instaladas)
3. Execute **Run All** ou célula por célula na ordem

### 4.2 Tempo de execução esperado

| Seção                          | Tempo Estimado |
|--------------------------------|----------------|
| Imports e carregamento         | ~2s            |
| Limpeza                        | ~1s            |
| EDA (gráficos)                 | ~3s            |
| Simulação                      | ~2s            |
| Treinamento dos modelos        | ~30-60s        |
| Importância das features       | ~15s           |
| Previsão + simulação temporal  | ~5s            |
| **Total**                      | **~1-2 min**   |

---

## 5. Como Adaptar para Dados Reais

Quando múltiplas inspeções estiverem disponíveis, siga estes passos:

### 5.1 Preparar os dados

Organize os arquivos de inspeção com identificação do ano:

```
Tcc/
├── inspecao_2010.xlsx
├── inspecao_2015.xlsx
├── inspecao_2020.xlsx
```

### 5.2 Parear anomalias entre inspeções

Para calcular a taxa de corrosão real, é necessário identificar a **mesma anomalia** em inspeções diferentes. Critérios de pareamento sugeridos:

```python
# Critério: mesmo ID do tubo + posição próxima (tolerância de ±0.5m)
tolerancia_m = 0.5

for anomalia_nova in inspecao_2015:
    match = inspecao_2010[
        (inspecao_2010['id_tubo'] == anomalia_nova['id_tubo']) &
        (abs(inspecao_2010['posicao_m'] - anomalia_nova['posicao_m']) < tolerancia_m)
    ]
    if len(match) == 1:
        taxa_real = (anomalia_nova['profundidade_pct'] - match['profundidade_pct']) / 5
```

### 5.3 Calcular a taxa de corrosão real

```python
df_pareado['taxa_corrosao_pct_ano'] = (
    (df_pareado['profundidade_nova'] - df_pareado['profundidade_anterior'])
    / intervalo_anos
)
```

### 5.4 Substituir a simulação

No notebook, **substitua a Seção 4** (simulação) pelo carregamento dos dados pareados reais. O restante do pipeline (features, treinamento, avaliação) permanece o mesmo.

---

## 6. Interpretação dos Resultados

### 6.1 Métricas

| Métrica | O que mede | Como interpretar |
|---------|-----------|-----------------|
| **R²**  | Variância explicada (0 a 1) | Acima de 0.80 é bom para dados de corrosão |
| **MAE** | Erro médio absoluto (%/ano) | Quanto menor, melhor. Compare com a taxa média |
| **RMSE**| Erro quadrático médio (%/ano) | Penaliza erros grandes. Deve ser próximo do MAE |

### 6.2 Feature Importance

As features com maior importância são as que mais influenciam a previsão:
- Use para entender os **fatores de risco** de corrosão acelerada
- Features irrelevantes (importância ~0) podem ser removidas para simplificar o modelo

### 6.3 Gráfico de Evolução Temporal

O gráfico da Seção 9 mostra como a profundidade de uma anomalia progride ao longo dos anos:
- **Linha vermelha**: trajetória prevista
- **Linha laranja tracejada (80%)**: limite crítico típico para reparo
- O cruzamento indica o **ano estimado para intervenção**

---

## 7. Solução de Problemas

| Problema | Solução |
|----------|---------|
| `ModuleNotFoundError: No module named 'seaborn'` | Verifique se o kernel do notebook usa o mesmo Python onde instalou as libs. Reinstale com `pip install seaborn` |
| `ModuleNotFoundError: No module named 'xgboost'` | `pip install xgboost` |
| Kernel não encontrado | `pip install ipykernel && python -m ipykernel install --user` |
| Dados com formato numérico errado | A função `limpar_numero()` trata o formato BR. Se os dados estiverem em formato US, remova a limpeza |
| Treinamento muito lento | Reduza `n_estimators` para 100 nos modelos Random Forest e XGBoost |
