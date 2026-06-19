# State of Data Brasil — Desigualdades de Gênero (2021–2024)

Análise longitudinal das edições 2021–2024 do **State of Data Brasil**, investigando como gênero se relaciona com remuneração, senioridade e acesso a cargos de liderança no mercado de dados brasileiro. O projeto combina EDA, inferência estatística, modelagem preditiva com análise de equidade (Fairlearn) e explicabilidade via SHAP.

---

## Perguntas de Pesquisa

**P1 — Comportamental (2023–2024):** Homens e mulheres adotam ferramentas de IA generativa de formas diferentes? Em que tipo de uso e contextos?

**P2 — Estrutural (2021–2024):** A composição de cargos, distribuição salarial e acesso à senioridade mudaram ao longo dos quatro anos, e essa mudança afeta os gêneros de forma distinta?

---

## Estrutura do Repositório

```
stateofdata/
├── data/
│   ├── state_of_data_2021.csv
│   ├── state_of_data_2022.csv
│   ├── state_of_data_2023.csv
│   ├── state_of_data_2024.csv
│   ├── df_train.parquet          # 2021–2023 (gerado pela Seção 1)
│   └── df_test.parquet           # 2024 (gerado pela Seção 1)
├── figures/                      # Gráficos exportados (PNG, DPI 150)
├── state_of_data_genero.ipynb    # Notebook principal
├── df_full.xlsx                  # Dataset completo pós-processamento
├── resultados_estatisticos.xlsx  # Tabelas dos testes estatísticos
└── README.md
```

---

## Pré-requisitos

Python 3.9+

```bash
pip install pandas numpy plotly matplotlib seaborn scipy statsmodels \
            scikit-learn xgboost shap fairlearn openpyxl pyarrow nbformat
```

---

## Visão Geral do Notebook

| Seção | Título | Descrição |
|-------|--------|-----------|
| **0** | Configurações | Imports, constantes, mapeamentos, engenharia de features de IA e paleta de cores |
| **1** | Pré-processamento | Pipeline de limpeza, unificação dos 4 anos e criação de targets salariais agrupados |
| **2** | EDA | 10 visualizações interativas (Plotly): gênero e regional |
| **3** | Estatística Inferencial | Hipóteses H1–H6 com qui-quadrado, Mann-Whitney, Spearman, regressão logística |
| **4** | Modelagem (ML) | Treino/CV/avaliação com análise de equidade por gênero via Fairlearn |
| **5** | Explicabilidade (SHAP) | Importância de features global e por gênero, com destaque para variáveis de IA |

---

## Seção 0 — Configurações

Define toda a infraestrutura de mapeamento e engenharia de features:

- **`COLUMN_MAPPING`** — normaliza nomes de colunas entre os 4 anos (esquemas divergem entre edições)
- **`IA_COLS_RAW`** — colunas brutas de IA generativa disponíveis em 2023 e 2024
- **`engenharia_ia()`** — cria 7 features binárias (`ia_*`) a partir das colunas brutas: uso individual (não usa / gratuita / paga bolso próprio / paga pela empresa / Copilot), prioridade estratégica da empresa e uso difuso na organização
- **`IA_FEATURES_MODELO_B`** — lista das 7 features de IA usadas no Modelo B
- **`FAIXA_SALARIAL_MAP / ORDEM`** — padronização e ordenação ordinal das 12 faixas (R$1k a R$40k+)
- **`CARGO_MAP`** — consolida ~30 variações em 8 categorias (Analista, Engenheiro, Cientista de Dados, etc.)
- **`AREA_FORMACAO_MAP`, `REGIAO_MAP`, `SITUACAO_MAP`, `MODALIDADE_MAP`** — normalizações auxiliares
- **Constantes globais:** `SEED = 42`, `ALPHA = 0.05`, paletas de cores por gênero e região

---

## Seção 1 — Pré-processamento

Pipeline modular com quatro funções:

- **`build_year_dataframe(df_raw, year)`** — seleciona e renomeia colunas para esquema unificado; injeta features de IA em 2023/2024
- **`preprocess(df)`** — aplica transformações por variável (ver tabela abaixo)
- **`criar_targets_salariais(df)`** — cria versões agrupadas da faixa salarial para experimentos de sensibilidade
- **`run_pipeline(dfs_raw)`** — executa as etapas para todos os anos e exporta os artefatos

| Variável | Transformação |
|----------|--------------|
| `genero` | Filtra apenas "Masculino" e "Feminino" |
| `faixa_salarial` | Padroniza texto e converte para Categorical ordinal (12 faixas) |
| `faixa_salarial_3` | Agrupamento em 3 classes: Baixa / Média / Alta |
| `faixa_salarial_bin` | Binário: Alta (≥ R$12k) vs. demais |
| `nivel` | Categorical ordinal: Júnior → Pleno → Sênior (Gestor excluído do ML — ver decisões metodológicas) |
| `gestor` | Binarizado: Sim → 1, Não → 0 |
| `nivel_ensino` | Categorical ordinal (6 níveis) |
| `cargo` | Consolidado via `CARGO_MAP` (8 categorias) |
| `regiao` | Normalizado; "Exterior" e "Prefiro não informar" removidos |
| `situacao` | Categorias consolidadas via `SITUACAO_MAP` |
| `modalidade` | Normalizado via `MODALIDADE_MAP` |

**Artefatos exportados:** `df_full.parquet` (completo), `df_train.parquet` (2021–2023), `df_test.parquet` (2024).

---

## Seção 2 — EDA

10 visualizações interativas (Plotly), sem redundância entre si.

| Subseção | Pergunta |
|----------|----------|
| 2.1 | Evolução da representatividade feminina (2021–2024) |
| 2.2 | Distribuição de senioridade por gênero e ano |
| 2.3 | Distribuição de cargos por gênero |
| 2.4 | Diferença salarial entre gêneros |
| 2.5 | Nível de escolaridade por gênero |
| 2.6 | Área de formação predominante por gênero |
| 2.7 | Tempo de experiência em dados por gênero |
| 2.8 | Modalidade de trabalho por gênero |
| 2.9 | Adoção de ferramentas de IA generativa por gênero |
| 2.10 | Representatividade feminina comparada entre as 5 regiões brasileiras |

---

## Seção 3 — Estatística Inferencial

Seis hipóteses testadas; todos os resultados exportados para `resultados_estatisticos.xlsx`.

| Hipótese | Teste | Métrica de efeito |
|----------|-------|-------------------|
| H1/H2 — Associação gênero × cargo/nível/região/formação | Qui-quadrado | V de Cramér |
| H3 — Diferença salarial entre gêneros | Mann-Whitney U (two-sided) | rank-biserial r |
| H4 — Senioridade/experiência × remuneração | Correlação de Spearman + Fisher's z | ρ (rho) |
| H5 — Tendência temporal da participação feminina | Regressão logística + Cochran-Armitage | OR, IC 95%, McFadden R² |
| H6 — Adoção de IA por gênero (2023–2024) | Qui-quadrado + IC de Wilson | V de Cramér, gap em pp |

---

## Seção 4 — Modelagem Preditiva

### Variáveis-alvo

| Target | Tipo | Descrição |
|--------|------|-----------|
| `faixa_salarial` | Multiclasse ordinal | 12 faixas de R$1k a R$40k+ |
| `faixa_salarial_3` | Multiclasse | 3 classes agrupadas: Baixa / Média / Alta |
| `faixa_salarial_bin` | Binária | Alta (≥ R$12k) vs. demais |
| `nivel` | Multiclasse ordinal | Júnior · Pleno · Sênior |
| `gestor` | Binária | Atua ou não como gestor/a |

Os experimentos centrais utilizam `TARGETS_CENTRAIS = ['gestor', 'nivel', 'faixa_salarial_bin']` com o estimador XGBoost.

### Dois modelos comparados

| Modelo | Features | Objetivo |
|--------|----------|---------|
| **A** (baseline) | 9 features estruturais | Estrutura base do mercado |
| **B** | Features estruturais + 7 features de IA (`ia_*`) | Mede se uso de IA generativa é preditor relevante |

### Algoritmos e seleção

Três algoritmos avaliados por target: **Logistic Regression**, **Random Forest** (200 estimadores) e **XGBoost** (200 estimadores). Seleção via cross-validation estratificado (5 folds, F1-macro). O melhor modelo em CV é retreinado no conjunto completo de treino e avaliado em 2024.

### Pré-processamento das features

`ColumnTransformer` com imputação mediana + `StandardScaler` para numéricas, e imputação moda + `OneHotEncoder` (`handle_unknown='ignore'`) para categóricas.

### Análise de equidade (Fairlearn)

Para cada target central, a análise de equidade reporta:

- **F1-macro por gênero** — identifica viés de desempenho entre grupos
- **`demographic_parity_difference`** — diferença na taxa de predição positiva entre gêneros
- **`equalized_odds_difference`** — diferença nas taxas de verdadeiro positivo e falso positivo entre gêneros

---

## Seção 5 — Explicabilidade (SHAP)

### 5.1 SHAP global: Modelo A vs. Modelo B

Para cada target central (Modelo A e B):

1. Transforma o conjunto de teste com o pré-processador treinado
2. Amostra até 200 observações (performance)
3. Calcula SHAP via `TreeExplainer` (RF/XGBoost)
4. Para multiclasse, agrega pela média dos valores absolutos entre classes e normaliza para soma = 1
5. Plota top-15 features; no Modelo B, features de IA são destacadas em laranja

A função `plot_shap_bar` gera painéis lado a lado (A vs. B) para comparação direta do impacto das features de IA.

### 5.2 SHAP por gênero + correlação de Spearman entre rankings

Para cada target central:

1. Calcula SHAP separadamente para respondentes masculinos e femininos
2. Plota top-12 features para cada grupo (cores distintas por gênero; features de IA em laranja)
3. Computa correlação de Spearman entre os rankings de importância dos dois grupos, reportando ρ, p-valor e interpretação qualitativa (Alta / Moderada / Baixa concordância)



