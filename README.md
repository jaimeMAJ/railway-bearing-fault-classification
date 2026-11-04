# Railway Bearing Fault Classification

Classificação de falhas em rolamentos ferroviários via Machine Learning, usando processamento de sinais de vibração (estatísticas temporais e espectrais via FFT/Envelope) e o dataset CWRU como benchmark. Trabalho de Conclusão de Curso — Engenharia Mecatrônica, UFSJ.

## Sobre o projeto

Este projeto implementa um pipeline de diagnóstico e classificação de falhas em rolamentos a partir de sinais de vibração acelerométrica, cobrindo desde a ingestão de dados brutos até a validação cruzada agrupada e avaliação por votação majoritária de ensaio. Como benchmark inicial, utiliza o dataset público [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter), permitindo validar a metodologia antes de aplicá-la a dados reais de rodeiros ferroviários.

**Classes de integridade física classificadas:**
- `Normal` — rolamento saudável (baseline)
- `IR` — falha na pista interna (Inner Race Fault)
- `B` — falha em elemento rolante (Ball Fault)
- `OR` — falha na pista externa, posição @6:00 (Outer Race Fault)

Cada classe de defeito possui 3 níveis de severidade (0.007", 0.014", 0.021") e 4 condições de carga do motor (0 a 3 HP).

## Pipeline Metodológico

1. **Download Automatizado** (`download_cwru.py`) — baixa os arquivos `.mat` (12 kHz, Drive End) do site oficial da CWRU com retry automático.
2. **Ingestão e Segmentação com Linhagem** (`src/preprocessing.py`) — lê os arquivos `.mat`, extrai metadados, imputa RPM ausente com base na carga e segmenta os sinais contínuos em janelas discretas ($N = 2048$ pontos), preservando rastreabilidade de amostras (*data lineage*).
3. **Extração de Características** (`src/features.py`) — calcula atributos estocásticos temporais (RMS, Curtose, Fator de Crista, Desvio Padrão, Assimetria, Pico-a-Pico, Fator de Forma, Fator de Impulso) e atributos espectrais (FFT com janela de Hanning, Análise de Envelope via Hilbert com remoção de DC, energias em torno de BPFO, BPFI, BSF, FTF e centroide espectral).
4. **Feature Store** (`01_extract_features.py`) — valida a qualidade dos dados (0 nulos, 0 infinitos) e persiste a base em formato colunar Apache Parquet (`features.parquet`) e CSV (`features.csv`).
5. **Modelagem Preditiva e Avaliação** (`02_train_models.py` e `src/models.py`) — divisão treino/teste estratificada por arquivo de ensaio (70/30) para evitar vazamento temporal de dados (*data leakage*), validação cruzada agrupada (*Stratified Group K-Fold*, $k=4$), classificação via Random Forest e SVM (com StandardScaler), matrizes de confusão (300 DPI), ranking de importância física de features e votação majoritária por ensaio.
6. **Visualização Espectral Acadêmica** (`03_plot_spectra.py`) — gráficos comparativos de formas de onda, FFT e espectros de envelope com frequências teóricas de falha para a monografia.

## Estrutura do Repositório

```text
notebook_python/
├── data/
│   ├── raw/                  # Arquivos .mat originais do CWRU (não versionado)
│   └── processed/            # Feature Store persistida (features.parquet e features.csv)
├── notebooks/                # Exploração interativa de sinais e experimentos
├── src/
│   ├── __init__.py
│   ├── config.py             # Parâmetros físicos do rolamento SKF 6205, taxa e caminhos
│   ├── preprocessing.py      # Ingestão, metadados, sliding window e data lineage
│   ├── features.py           # Extração estocástica (tempo) e espectral (FFT/envelope)
│   └── models.py             # Pipelines RF/SVM, Group K-Fold, métricas e visualizações
├── results/
│   ├── figures/              # Gráficos acadêmicos em alta resolução (300 DPI) para monografia
│   ├── cv_resultados.csv     # Resultados de validação cruzada por fold e subconjunto
│   └── votacao_ensaios_rf.csv# Desempenho do diagnóstico em nível de rolamento (file-level)
├── 01_extract_features.py    # Pipeline de ingestão e geração da Feature Store
├── 02_train_models.py        # Treino, validação cruzada agrupada e avaliação final
├── 03_plot_spectra.py        # Geração de espectros FFT e Envelope para a monografia
├── download_cwru.py          # Script de download automatizado do dataset CWRU
├── requirements.txt          # Dependências do projeto (incluindo pyarrow)
└── README.md
```

## Como Reproduzir

```powershell
# 1. Clonar o repositório
git clone https://github.com/jaimeMAJ/railway-bearing-fault-classification.git
cd railway-bearing-fault-classification

# 2. Criar e ativar o ambiente virtual
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Baixar o dataset CWRU
python download_cwru.py

# 5. Executar a extração de features e geração da Feature Store
python 01_extract_features.py

# 6. Treinar os modelos e calcular a validação cruzada
python 02_train_models.py

# 7. Gerar os espectros e figuras de alta resolução para o TCC (300 DPI)
python 03_plot_spectra.py
```

## Resultados Atualizados

Os modelos foram avaliados com divisão treino/teste estratificada por arquivo de ensaio (70/30) para prevenir estritamente o vazamento de dados (*data leakage*), além de validação cruzada agrupada em 4 folds (*Stratified Group K-Fold*).

### Desempenho no Conjunto de Teste (886 janelas, 12 arquivos independentes)

#### Random Forest (Conjunto Completo de Features: Tempo + Frequência)

| Classe | Precision | Recall | F1-score | Suporte |
|---|---|---|---|---|
| B (elemento rolante) | 0.99 | 0.97 | 0.98 | 177 |
| IR (pista interna) | 1.00 | 1.00 | 1.00 | 236 |
| Normal (saudável) | 0.99 | 1.00 | 0.99 | 237 |
| OR (pista externa) | 1.00 | 1.00 | 1.00 | 236 |

* **Acurácia Geral por Janela:** **99.32%**
* **Acurácia em Nível de Ensaio (Votação Majoritária):** **100.00%** (12 de 12 ensaios classificados com acerto pleno)

#### SVM (StandardScaler + Kernel RBF)
* **Acurácia Geral por Janela:** **99.10%**
* **F1-Score Macro:** **0.9910**

### Comparativo de Validação Cruzada Agrupada (Stratified Group 4-Fold)

| Modelo | Subconjunto de Features | Acurácia Média | Desvio Padrão | F1-Score Macro |
|---|---|---|---|---|
| **Random Forest** | `todas_features` (16) | **99.41%** | $\pm 0.46\%$ | **0.9942** |
| **Random Forest** | `frequencia_envelope` (8) | **99.48%** | $\pm 0.46\%$ | **0.9947** |
| **Random Forest** | `tempo_completo` (8) | 94.98% | $\pm 1.75\%$ | 0.9462 |
| **SVM (RBF)** | `todas_features` (16) | **99.43%** | $\pm 0.50\%$ | **0.9943** |
| **SVM (RBF)** | `frequencia_envelope` (8) | 95.10% | $\pm 7.52\%$ | 0.9423 |
| **SVM (RBF)** | `tempo_base` (3) | 91.52% | $\pm 7.04\%$ | 0.9105 |

### Importância Física dos Atributos (Random Forest — MDI)
1. **Desvio Padrão ($\sigma$):** $17.29\%$
2. **Centroide Espectral:** $16.55\%$
3. **Pico a Pico:** $15.06\%$
4. **Valor Eficaz (RMS):** $12.53\%$
5. **Energia BPFI (Envelope):** $11.29\%$

## Próximos Passos

- [x] Testar classificador SVM e comparar com Random Forest
- [x] Adicionar features no domínio da frequência (FFT e Análise de Envelope via Hilbert)
- [x] Validação cruzada estratificada agrupada por ensaio (*Stratified Group K-Fold*)
- [x] Implementar avaliação por votação majoritária de ensaio (*File-level Voting*)
- [ ] Modelagem expandida para 10 classes (diferenciando por severidade do dano mecânico)
- [ ] Aplicação da metodologia a dados reais de rodeiros ferroviários

## Autor

Jaime Junior — Engenharia Mecatrônica, UFSJ  
Orientadora: Profa. Mariana Geny Moreira Vale Cardoso
