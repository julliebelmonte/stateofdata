# State of Data Brasil — Análise de Desigualdades de Gênero (2021–2024)

Análise longitudinal das edições 2021, 2022, 2023 e 2024 do **State of Data Brasil**, com foco nas desigualdades de gênero no mercado de dados. O projeto combina análise exploratória (EDA), inferência estatística, modelagem preditiva com Machine Learning e explicabilidade via SHAP.

---

## Objetivo

Investigar como gênero se relaciona com **remuneração**, **senioridade** e **acesso a cargos de liderança** no mercado de dados brasileiro, observando tendências ao longo de quatro anos e avaliando se o uso de IA generativa (medido a partir de 2023) é um preditor relevante dessas variáveis.

---

## Estrutura do Repositório

```
stateofdata/
├── data/
│   ├── state_of_data_2021.csv
│   ├── state_of_data_2022.csv
│   ├── state_of_data_2023.csv
│   ├── state_of_data_2024.csv
│   ├── df_train.parquet          # 2021–2023, gerado pela Seção 1
│   └── df_test.parquet           # 2024, gerado pela Seção 1
├── figures/                      # Gráficos exportados (PNG)
├── state_of_data_genero.ipynb    # Notebook principal (único ponto de entrada)
├── df_full.xlsx                  # Dataset completo pós-processamento
├── resultados_estatisticos.xlsx  # Tabelas dos testes estatísticos
└── README.md
```

> **Atenção:** Execute sempre a **Seção 1 (Pré-processamento)** antes de qualquer outra. As demais seções dependem dos arquivos `df_train.parquet` e `df_test.parquet` que ela exporta.

---

## Pré-requisitos

**Python 3.9 ou superior** é recomendado.

```bash
pip install pandas numpy plotly matplotlib seaborn scipy statsmodels \
            scikit-learn xgboost shap openpyxl pyarrow nbformat
```

---

## Como Executar

1. Clone o repositório e coloque os quatro CSVs brutos em `data/`
2. Abra `state_of_data_genero.ipynb` no Jupyter Lab ou VS Code
3. Execute as seções em ordem sequencial, começando pela Seção 1

---

## Visão Geral do Notebook

O notebook está organizado em **6 seções principais**:

| Seção | Título | Descrição resumida |
|-------|--------|--------------------|
| **0** | Configurações | Imports, constantes, mapeamentos e paleta de cores |
| **1** | Pré-processamento | Pipeline de limpeza e unificação dos 4 anos |
| **2** | EDA — Gênero e Regional | 10 visualizações interativas (Plotly) |
| **3** | Estatística Inferencial | Qui-quadrado, Mann-Whitney, regressão logística |
| **4** | Modelagem (Machine Learning) | Treino, CV, avaliação de equidade por gênero |
| **5** | Explicabilidade (SHAP) | Importância de features com destaque para IA |

---

## Seção 0 — Configurações

Define toda a infraestrutura de mapeamento do projeto:

- **`COLUMN_MAPPING`**: dicionário que normaliza os nomes de colunas entre os 4 anos da pesquisa (as edições têm esquemas diferentes)
- **`IA_COLS`**: mapeamento das features de IA generativa disponíveis apenas em 2023 e 2024
- **`FAIXA_SALARIAL_MAP` / `FAIXA_SALARIAL_ORDEM`**: padronização e ordenação ordinal das 12 faixas salariais (de R$1k a R$40k+)
- **`CARGO_MAP`**: consolidação de ~30 variações de cargos em 8 categorias principais (Analista de Dados, Engenheiro de Dados, Cientista de Dados, etc.)
- **`AREA_FORMACAO_MAP`**, **`REGIAO_MAP`**, **`SITUACAO_MAP`**: normalizações auxiliares
- **Constantes globais**: `SEED = 42`, `ALPHA = 0.05`, paletas de cores por gênero e região

---

## Seção 1 — Pipeline de Pré-processamento

Pipeline modular composto por três funções principais:

### `build_year_dataframe(df_raw, year)`
Seleciona e renomeia as colunas de cada ano para um esquema unificado. Para 2023 e 2024, também injeta as colunas de IA generativa (`ia_prioridade_empresa`, `ia_tipos_uso_gestao`, `ia_motivos_nao_usar`, `ia_tipo_uso_pessoal`, `ia_usa_chatgpt_no_trabalho`).

### `preprocess(df)`
Aplica as seguintes transformações:

| Variável | Transformação |
|----------|--------------|
| `genero` | Filtra apenas "Masculino" e "Feminino" |
| `nivel_ensino` | Converte para Categorical ordinal (6 níveis) |
| `faixa_salarial` | Padroniza texto, mapeia abreviações, converte para Categorical ordinal (12 faixas) |
| `cargo` | Consolida variações via `CARGO_MAP` (8 categorias) |
| `area_formacao` | Consolida via `AREA_FORMACAO_MAP` |
| `regiao` | Normaliza nomes, remove "Exterior" e "Prefiro não informar" |
| `nivel` | Categorical ordinal: Júnior → Pleno → Sênior (Gestor excluído por leakage no ML) |
| `gestor` | Binariza: Sim → 1, Não → 0 |
| `situacao` | Consolida categorias, remove valores nulos |

### `run_pipeline(dfs_raw)`
Executa as etapas acima para todos os anos e concatena em um único DataFrame. Ao final, exporta:
- `data/df_full.parquet` — dataset completo (2021–2024)
- `data/df_train.parquet` — anos 2021–2023 (treino ML)
- `data/df_test.parquet` — ano 2024 (teste ML)

**Cobertura das features de IA:** o notebook reporta a % de respondentes com pelo menos uma feature de IA preenchida no treino e no teste.

---

## Seção 2 — Análise Exploratória de Dados (EDA)

10 visualizações interativas geradas com **Plotly**, organizadas em duas subseções:

### 2.1–2.9 — EDA por Gênero (longitudinal, 2021–2024)

Cada gráfico responde uma pergunta específica, sem redundância:

| Subseção | Pergunta respondida |
|----------|-------------------|
| 2.1 | Como evoluiu a representatividade feminina ao longo dos anos? |
| 2.2 | Qual a distribuição de senioridade por gênero em cada ano? |
| 2.3 | Como se distribui o tipo de cargo por gênero? |
| 2.4 | Qual a diferença salarial entre gêneros? |
| 2.5 | Qual o nível de escolaridade por gênero? |
| 2.6 | Qual a área de formação predominante por gênero? |
| 2.7 | Como se distribui o tempo de experiência em dados por gênero? |
| 2.8 | Qual a modalidade de trabalho (remoto/híbrido/presencial) por gênero? |
| 2.9 | Como homens e mulheres adotam ferramentas de IA generativa? |

Os gráficos utilizam dois layouts:
- **`subplots_pct_fem`**: barras horizontais com % feminino por categoria, em subplots 1×4 (um por ano)
- **`subplots_piramide`**: pirâmide bilateral comparando a distribuição interna de cada gênero

### 2.10 — EDA Regional

4 gráficos comparando a representatividade feminina entre as 5 regiões brasileiras (Sudeste, Sul, Nordeste, Centro-Oeste e Norte).

---

## Seção 3 — Estatística Inferencial

Todos os resultados são exportados para `resultados_estatisticos.xlsx` com uma aba por tipo de teste.

### Testes Utilizados

| Teste | Hipótese | Métrica de efeito |
|-------|----------|-------------------|
| **Qui-quadrado + V de Cramér** | Associação entre gênero e variáveis categoriais (cargo, nível, região, área de formação) | V de Cramér (tamanho do efeito) |
| **Mann-Whitney U** | Diferença na distribuição salarial entre gêneros | p-valor, U |
| **Correlação de Spearman** | Relação entre nível de senioridade e faixa salarial; experiência e faixa salarial | ρ (rho), p-valor |
| **Regressão logística simples** | Tendência temporal da proporção feminina (2021–2024), nacional e por região | OR, IC 95%, McFadden R² |

### Saídas Estatísticas

| Aba no Excel | Conteúdo |
|---|---|
| `Chi2 e Cramer V` | Resultados por variável categorial e por ano |
| `Mann-Whitney Salarial` | Diferença salarial por gênero, por ano |
| `Spearman Nivel x Salario` | Correlação nível × salário por gênero e ano |
| `Spearman Exp x Salario` | Correlação experiência × salário |
| `Tendencia Temporal` | Regressão logística: tendência da representatividade feminina |

---

## Seção 4 — Modelagem Preditiva (Machine Learning)

### Variáveis-Alvo

| Target | Tipo | Descrição |
|--------|------|-----------|
| `faixa_salarial` | Multiclasse ordinal | 12 faixas de R$1k a R$40k+ |
| `nivel` | Multiclasse ordinal | Júnior · Pleno · Sênior |
| `gestor` | Binária | Atua ou não como gestor/a |

> `gestor` é tratado separadamente de `nivel` para evitar leakage (as variáveis são altamente correlacionadas).

### Features Base

```
nivel_ensino, area_formacao, cargo, tempo_area_dados, tempo_area_ti,
modalidade, setor, regiao, situacao
```

### Dois Modelos Comparados

| Modelo | Treino | Teste | Features extras |
|--------|--------|-------|-----------------|
| **Modelo A** | 2021–2023 | 2024 | Nenhuma (baseline) |
| **Modelo B** | 2021–2023 (c/ IA 2023) | 2024 (c/ IA 2024) | `ia_*` (5 features) |

> A comparação A vs B responde: **"O uso de IA generativa é um preditor relevante de salário e senioridade?"**

### Algoritmos e Seleção

Três algoritmos são avaliados para cada target:

- **Logistic Regression** — `max_iter=500`, `class_weight='balanced'`
- **Random Forest** — 200 estimadores, `class_weight='balanced'`
- **XGBoost** — 200 estimadores, `scale_pos_weight=5` para target binário

**Seleção:** cross-validation estratificado com 5 folds, métrica **F1-macro**. O melhor modelo em CV é retreinado no treino completo e avaliado no teste.

### Análise de Equidade (Fairness)

O desempenho no conjunto de teste é reportado **separadamente por gênero** (F1-macro para Masculino e Feminino), permitindo identificar se os modelos têm viés de desempenho entre os grupos.

### Pré-processamento das Features

Um `ColumnTransformer` é construído automaticamente:
- **Numéricas**: imputação pela mediana + StandardScaler
- **Categóricas**: imputação pela moda + OneHotEncoder (`handle_unknown='ignore'`)

---

## Seção 5 — Explicabilidade com SHAP

### Estratégia

Para cada combinação de (Modelo A/B) × (target), o notebook:

1. Transforma o conjunto de teste com o pré-processador treinado
2. Amostra até 600 observações (para performance)
3. Calcula SHAP values usando `TreeExplainer` (Random Forest / XGBoost) ou `LinearExplainer` (Logistic Regression)
4. Para classificação multiclasse, agrega pela **média dos valores absolutos** entre as classes
5. Gera gráfico de barras com as top-15 features por importância

### Destaque Visual

No **Modelo B**, features de IA (prefixo `ia_`) são destacadas em **laranja** (`#F4A43B`), enquanto features base ficam em azul. Isso torna imediatamente visível se e onde a IA generativa entra como fator relevante.

### Gráfico Comparativo

A função `plot_shap_comparacao` gera um painel lado a lado (Modelo A vs Modelo B) para cada target, facilitando a comparação direta do impacto da inclusão das features de IA.

### Arquivos Exportados

```
figures/
├── shap_A_faixa_salarial.png   # Modelo A — Faixa Salarial
├── shap_B_faixa_salarial.png   # Modelo B — Faixa Salarial
├── shap_A_nivel.png            # Modelo A — Senioridade
├── shap_B_nivel.png            # Modelo B — Senioridade
├── shap_A_gestor.png           # Modelo A — Gestor/a
├── shap_B_gestor.png           # Modelo B — Gestor/a
├── shap_comp_faixa_salarial.png  # Comparativo A vs B
├── shap_comp_nivel.png           # Comparativo A vs B
└── shap_comp_gestor.png          # Comparativo A vs B
```

---

## Decisões Metodológicas Relevantes

| Decisão | Justificativa |
|---------|--------------|
| Faixa salarial como classificação ordinal (não regressão) | O State of Data coleta salário em faixas, não valor contínuo |
| `Gestor` removido de `NIVEL_SENIORIDADE_ORDEM` no ML | Evita leakage com o target `gestor` |
| Split temporal (treino 2021–2023 / teste 2024) | Simula generalização real: modelos treinados no passado predizem o futuro |
| Features de IA separadas em Modelo A/B | Isola o efeito das variáveis de IA na performance preditiva |
| F1-macro como métrica principal | Robusta a desbalanceamento de classes |
| `class_weight='balanced'` / `scale_pos_weight` | Mitiga o desbalanceamento, especialmente no target `gestor` |

---

## Pontos de Atenção / Trabalho em Andamento

- **Leakage no Modelo B para o target `gestor`**: colunas de IA podem estar preenchidas predominantemente por respondentes em cargos de gestão (padrão de resposta diferenciado). Requer investigação e ajuste.
- A análise de **model drift** entre anos (prevista na proposta original) ainda não está implementada.

---

## Outputs Gerados

| Arquivo | Conteúdo |
|---------|----------|
| `data/df_full.parquet` | Dataset unificado pós-processamento (2021–2024) |
| `data/df_train.parquet` | Split de treino (2021–2023) |
| `data/df_test.parquet` | Split de teste (2024) |
| `df_full.xlsx` | Versão Excel do dataset completo |
| `resultados_estatisticos.xlsx` | Tabelas de todos os testes estatísticos (5 abas) |
| `figures/` | Todos os gráficos EDA e SHAP em PNG (DPI 150) |
