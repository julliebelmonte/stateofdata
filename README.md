# State of Data Brasil — Desigualdades de Gênero (2021–2024)

Análise longitudinal das edições 2021–2024 do **State of Data Brasil**, investigando como gênero se relaciona com remuneração, senioridade e acesso a cargos de liderança no mercado de dados brasileiro. O projeto combina EDA, inferência estatística, modelagem preditiva e explicabilidade via SHAP.

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

> **Atenção:** execute sempre a **Seção 1 (Pré-processamento)** antes das demais. As seções 2–5 dependem dos arquivos `df_train.parquet` e `df_test.parquet` que ela exporta.

---

## Pré-requisitos

Python 3.9+

```bash
pip install pandas numpy plotly matplotlib seaborn scipy statsmodels \
            scikit-learn xgboost shap openpyxl pyarrow nbformat
```

---

## Visão Geral do Notebook

| Seção | Título | Descrição |
|-------|--------|-----------|
| **0** | Configurações | Imports, constantes, mapeamentos e paleta de cores |
| **1** | Pré-processamento | Pipeline de limpeza e unificação dos 4 anos |
| **2** | EDA | 10 visualizações interativas (Plotly): gênero e regional |
| **3** | Estatística Inferencial | Hipóteses H1–H6 com qui-quadrado, Mann-Whitney, Spearman, regressão logística |
| **4** | Modelagem (ML) | Treino/CV/avaliação com análise de equidade por gênero |
| **5** | Explicabilidade (SHAP) | Importância de features com destaque para variáveis de IA |

---

## Seção 0 — Configurações

Define toda a infraestrutura de mapeamento:

- **`COLUMN_MAPPING`** — normaliza nomes de colunas entre os 4 anos (esquemas divergem entre edições)
- **`IA_COLS`** — mapeamento das features de IA generativa, disponíveis apenas em 2023 e 2024
- **`FAIXA_SALARIAL_MAP / ORDEM`** — padronização e ordenação ordinal das 12 faixas (R$1k a R$40k+)
- **`CARGO_MAP`** — consolida ~30 variações em 8 categorias (Analista, Engenheiro, Cientista de Dados, etc.)
- **`AREA_FORMACAO_MAP`, `REGIAO_MAP`, `SITUACAO_MAP`** — normalizações auxiliares
- **Constantes globais:** `SEED = 42`, `ALPHA = 0.05`, paletas de cores por gênero e região

---

## Seção 1 — Pré-processamento

Pipeline modular com três funções:

- **`build_year_dataframe(df_raw, year)`** — seleciona e renomeia colunas para esquema unificado; injeta features de IA em 2023/2024
- **`preprocess(df)`** — aplica transformações por variável (ver tabela abaixo)
- **`run_pipeline(dfs_raw)`** — executa as etapas para todos os anos e exporta os artefatos

| Variável | Transformação |
|----------|--------------|
| `genero` | Filtra apenas "Masculino" e "Feminino" |
| `faixa_salarial` | Padroniza texto e converte para Categorical ordinal (12 faixas) |
| `nivel` | Categorical ordinal: Júnior → Pleno → Sênior (Gestor excluído — ver decisões metodológicas) |
| `gestor` | Binarizado: Sim → 1, Não → 0 |
| `nivel_ensino` | Categorical ordinal (6 níveis) |
| `cargo` | Consolidado via `CARGO_MAP` (8 categorias) |
| `regiao` | Normalizado; "Exterior" e "Prefiro não informar" removidos |
| `situacao` | Categorias consolidadas via `SITUACAO_MAP` |

**Artefatos exportados:** `df_full.parquet` (completo), `df_train.parquet` (2021–2023), `df_test.parquet` (2024).

---

## Seção 2 — EDA

10 visualizações interativas (Plotly), sem redundância entre si.

### 2.1–2.9 — Por Gênero (longitudinal)

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

### 2.10 — Regional

Representatividade feminina comparada entre as 5 regiões brasileiras.

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
| `nivel` | Multiclasse ordinal | Júnior · Pleno · Sênior |
| `gestor` | Binária | Atua ou não como gestor/a |

### Dois modelos comparados

| Modelo | Features extras | Objetivo |
|--------|-----------------|----------|
| **A** (baseline) | Nenhuma | Estrutura base do mercado |
| **B** | 7 features de IA (`ia_*`) | Mede se uso de IA generativa é preditor relevante |

### Algoritmos e seleção

Três algoritmos avaliados por target: **Logistic Regression**, **Random Forest** (200 estimadores) e **XGBoost** (200 estimadores). Seleção via cross-validation estratificado (5 folds, F1-macro). O melhor modelo em CV é retreinado no conjunto completo de treino e avaliado em 2024.

### Pré-processamento das features

`ColumnTransformer` com imputação mediana + `StandardScaler` para numéricas, e imputação moda + `OneHotEncoder` (`handle_unknown='ignore'`) para categóricas.

### Análise de equidade

F1-macro reportado separadamente por gênero no conjunto de teste, permitindo identificar viés de desempenho entre grupos.

---

## Seção 5 — Explicabilidade (SHAP)

Para cada combinação (Modelo A/B) × target:

1. Transforma o conjunto de teste com o pré-processador treinado
2. Amostra até 300 observações (performance)
3. Calcula SHAP via `TreeExplainer` (RF/XGBoost) ou `LinearExplainer` (Logistic Regression)
4. Para multiclasse, agrega pela média dos valores absolutos entre classes e normaliza para soma = 1 (comparação proporcional entre modelos)
5. Plota top-15 features; no Modelo B, features de IA são destacadas em laranja

A função `plot_shap_comparacao` gera painéis lado a lado (A vs B) para comparação direta.

---

## Decisões Metodológicas

| Decisão | Justificativa |
|---------|--------------|
| Faixa salarial como classificação ordinal | Dados coletados em faixas, não valor contínuo |
| `Gestor` removido de `NIVEL_SENIORIDADE_ORDEM` no ML | Evita leakage com o target `gestor` |
| Split temporal treino 2021–2023 / teste 2024 | Simula generalização real: modelo treinado no passado prediz o futuro |
| Modelos A/B separados | Isola o efeito das variáveis de IA na performance preditiva |
| `class_weight='balanced'` / `scale_pos_weight` | Mitiga desbalanceamento de classes, especialmente em `gestor` |
| F1-macro como métrica principal | Robusta a desbalanceamento; penaliza igualmente erros em classes minoritárias |
| SHAP normalizado (soma = 1) | Torna a importância relativa comparável entre Modelo A e B |
| Fisher's z no Spearman (H4) | Testa se a correlação difere significativamente entre gêneros |
| Cochran-Armitage (H5) | Complementa a regressão logística detectando tendência monotônica sem assumir linearidade |

---

## Pontos de Atenção / Trabalho em Andamento

- **Possível leakage no Modelo B para o target `gestor`:** features de IA podem estar preenchidas predominantemente por respondentes em cargos de gestão (padrão de resposta diferenciado). Requer investigação.
- **Model drift entre anos** (previsto na proposta original) ainda não implementado.

---

## Outputs Gerados

| Arquivo | Conteúdo |
|---------|----------|
| `data/df_full.parquet` | Dataset unificado pós-processamento (2021–2024) |
| `data/df_train.parquet` | Split de treino (2021–2023) |
| `data/df_test.parquet` | Split de teste (2024) |
| `df_full.xlsx` | Versão Excel do dataset completo |
| `resultados_estatisticos.xlsx` | Tabelas de H1–H6 (6 abas) |
| `figures/` | Gráficos EDA e SHAP em PNG (DPI 150) |
