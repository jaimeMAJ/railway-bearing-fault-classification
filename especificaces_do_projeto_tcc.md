# Especificações do Projeto de TCC — Classificação de Falhas em Rolamentos Ferroviários

> **Instrução para o Agente de IA:** Este documento contém o escopo completo, a metodologia matemática e a arquitetura de software do Trabalho de Conclusão de Curso (TCC). Utilize estas diretrizes como fonte da verdade para gerar códigos em Python, estruturar pipelines de dados, fundamentar equações na escrita acadêmica e garantir que todas as entregas estejam alinhadas com os objetivos propostos.

---

## 1. Identificação do Projeto
* **Título:** Desenvolvimento de Modelo Preditivo Baseado em Inteligência Artificial para Classificação de Falhas em Rolamentos de Rodeiros Ferroviários
* **Instituição:** Universidade Federal de São João del-Rei (UFSJ)
* **Curso:** Engenharia Mecatrônica
* **Autor:** Jaime Junior
* **Orientadora:** Profa. Mariana Geny Moreira Vale Cardoso
* **Áreas de Atuação:** Engenharia Mecatrônica, Manutenção Preditiva (CBM - *Condition-Based Maintenance*), Processamento Digital de Sinais (PDS) e Aprendizado de Máquina (*Machine Learning*).
* **Duração Estimada:** 12 Semanas.

---

## 2. Contextualização e Objetivos

### 2.1 Problema de Engenharia
Rolamentos de eixos ferroviários (*axleboxes*) operam sob severas cargas dinâmicas, variações térmicas e impactos roda-trilho. A detecção tardia de falhas leva ao travamento do eixo e descarrilamentos. Métodos tradicionais (inspeção humana ou sensores térmicos de via) detectam anomalias apenas em estágios terminais. A análise de vibração aliada ao Aprendizado de Máquina permite diagnosticar defeitos precocemente em estágios iniciais de degradação.

### 2.2 Objetivo Geral
Desenvolver um framework computacional em Python para processamento de sinais de vibração e modelagem preditiva voltada à classificação multiclasse do tipo e severidade de falhas mecânicas em rolamentos, utilizando o banco de dados de referência da *Case Western Reserve University* (CWRU).

---

## 3. Especificações do Conjunto de Dados (Dataset)

* **Fonte:** CWRU Bearing Dataset (*Case Western Reserve University*).
* **Posição do Sensor:** Mancal da extremidade de carga (*Drive End* - DE).
* **Taxa de Amostragem ($f_s$):** $12\text{ kHz}$ ($12.000\text{ amostras/segundo}$).
* **Formato Original dos Arquivos:** MATLAB (`.mat`), contendo séries temporais de aceleração/vibração.
* **Estrutura de Classes (Classificação Multiclasse):**
  * **4 Condições Principais:**
    1. Saudável (*Normal / Healthy*)
    2. Falha na Pista Interna (*Inner Race Fault* - IR)
    3. Falha na Pista Externa (*Outer Race Fault* - OR)
    4. Falha nos Elementos Rolantes / Esferas (*Ball Fault* - B)
  * **3 Níveis de Severidade (Diâmetro do defeito induzido):**
    1. $0.007\text{ polegadas}$
    2. $0.014\text{ polegadas}$
    3. $0.021\text{ polegadas}$
  * **Mapeamento Alvo:** O agente deve apoiar tanto a modelagem de **4 classes** (agrupando por tipo de falha) quanto a de **10 classes** (Saudável + 3 tipos de falha $\times$ 3 severidades), conforme a evolução da análise exploratória.

---

## 4. Pipeline Técnico e Metodológico

O desenvolvimento do software deve ser modular e seguir estritamente as 4 etapas abaixo:

### Etapa 1: Ingestão e Pré-processamento de Dados
1. **Leitura:** Carregamento dos arquivos `.mat` utilizando `scipy.io.loadmat`, extraindo os vetores correspondentes ao sinal *Drive End* (ex: chaves terminadas em `_DE_time`).
2. **Filtragem de Ruído:** Aplicação opcional de filtros digitais (ex: passa-banda via `scipy.signal`) caso a análise exploratória indique ruído fora da banda de interesse.
3. **Segmentação (Sliding Window):**
   * Divisão dos sinais contínuos em janelas discretas de comprimento $N = 1024$ ou $N = 2048$ pontos.
   * Parametrização de sobreposição (*overlap*, ex: $0\%$ a $50\%$) para aumento de dados (*data augmentation*) e preservação de transientes nas bordas.
   * **Atenção Crítica contra *Data Leakage*:** A divisão entre conjuntos de treino e teste deve considerar blocos temporais distintos ou garantir que janelas sobrepostas de um mesmo instante não caiam simultaneamente no treino e no teste.

### Etapa 2: Engenharia de Atributos (Feature Engineering)
Para cada janela de sinal $x = [x_1, x_2, \dots, x_N]$, extrair um vetor estruturado de características nos domínios do tempo e da frequência:

#### A. Domínio do Tempo (Estatísticas Estocásticas)
* **Valor Médio Quadrático (RMS):** Mede a energia global do sinal.
  $$\text{RMS} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2}$$
* **Curtose (Kurtosis):** Mede a impulsividade do sinal (sensível a impactos periódicos de falhas incipientes).
  $$K = \frac{\frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^4}{\left(\frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^2\right)^2}$$
* **Fator de Crista (Crest Factor):** Razão entre o pico máximo e o valor RMS.
  $$\text{CF} = \frac{\max(|x_i|)}{\text{RMS}}$$
* **Atributos Complementares Recomendados:** Desvio Padrão ($\sigma$), Assimetria (*Skewness*), Valor de Pico a Pico (*Peak-to-Peak*), Fator de Forma (*Shape Factor*) e Fator de Impulso (*Impulse Factor*).

#### B. Domínio da Frequência (Análise Espectral)
* Aplicação de janela de suavização (ex: Hanning) seguida pela **Transformada Rápida de Fourier (FFT)** via `scipy.fft`.
* Extração de atributos espectrais:
  * Frequência de pico dominante e sua respectiva amplitude.
  * Energia espectral em bandas de frequência específicas.
  * Centroide espectral e variância espectral.
  * Monitoramento das frequências características de falha do rolamento (BPFO - *Ball Pass Frequency Outer*, BPFI - *Ball Pass Frequency Inner*, BSF - *Ball Spin Frequency*, FTF - *Fundamental Train Frequency*).

### Etapa 3: Modelagem de Aprendizado de Máquina (Machine Learning)
1. **Particionamento:** Divisão estratificada dos dados em **70% para Treinamento** e **30% para Teste** (`train_test_split` com `stratify=y`).
2. **Normalização:** Padronização das features (`StandardScaler` ou `MinMaxScaler` do `scikit-learn`) ajustada **apenas** nos dados de treino e aplicada ao teste.
3. **Modelos Preditivos:**
   * **Modelo Base:** *Random Forest Classifier* (`RandomForestClassifier`).
   * **Modelo Avançado:** *Support Vector Machines* (`SVC`), testando kernels linear e RBF.
4. **Otimização e Validação:**
   * Validação Cruzada Estratificada (*Stratified K-Fold Cross-Validation*, ex: $k = 5$ ou $10$).
   * Ajuste de hiperparâmetros via **Grid Search** (`GridSearchCV`):
     * *Random Forest:* `n_estimators`, `max_depth`, `min_samples_split`, `criterion`.
     * *SVM:* `C` (penalidade), `gamma`, `kernel`.

### Etapa 4: Avaliação e Validação de Resultados
Os modelos devem ser comparados utilizando métricas acadêmicas clássicas no conjunto de teste:
* **Matriz de Confusão** (visualização gráfica absoluta e normalizada).
* **Acurácia Global** (*Accuracy*).
* **Precisão** (*Precision*), **Revocação/Sensibilidade** (*Recall*) e **F1-Score** (média macro e ponderada por classe).
* Análise de importância de atributos (*Feature Importance* da Random Forest) para interpretabilidade física do diagnóstico mecatrônico.

---

## 5. Stack Tecnológica e Arquitetura de Repositório

### 5.1 Linguagem e Bibliotecas Principais
* **Linguagem:** Python 3.10+
* **Manipulação e Cálculo Numérico:** `numpy`, `pandas`
* **Processamento de Sinais e Leitura `.mat`:** `scipy` (`scipy.io`, `scipy.signal`, `scipy.fft`, `scipy.stats`)
* **Machine Learning:** `scikit-learn`
* **Visualização de Dados:** `matplotlib`, `seaborn`
* **Ambiente Interativo:** Jupyter Notebook (`.ipynb`) e scripts modulares (`.py`).

### 5.2 Estrutura de Diretórios Sugerida para o GitHub
```text
tcc-manutencao-preditiva-rolamentos/
│
├── data/
│   ├── raw/                  # Arquivos .mat originais do CWRU Dataset
│   └── processed/            # Datasets de features extraídas (.csv / .parquet)
│
├── notebooks/
│   ├── 01_exploracao_sinais.ipynb
│   ├── 02_extracao_features_fft.ipynb
│   ├── 03_treinamento_random_forest.ipynb
│   └── 04_treinamento_svm_comparacao.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Funções para ler arquivos .mat e aplicar sliding window
│   ├── feature_extraction.py # Funções matemáticas (RMS, Curtose, CF, FFT)
│   ├── models.py             # Pipelines do Scikit-Learn e GridSearchCV
│   └── evaluation.py         # Geração de matrizes de confusão e relatórios
│
├── results/
│   ├── figures/              # Gráficos de alta resolução (300 DPI) para a monografia
│   └── metrics/              # Tabelas de desempenho exportadas (.csv / .tex)
│
├── requirements.txt          # Dependências do projeto
└── README.md                 # Documentação principal do repositório
```

---

## 6. Cronograma de Execução (12 Semanas)

| Etapa | Atividade | Semanas Previstas | Entregável Técnico / Acadêmico |
| :--- | :--- | :---: | :--- |
| **1** | Revisão Bibliográfica e Estado da Arte | 1, 2, 3 | Capítulo de Fundamentação Teórica |
| **2** | Download e Estruturação do Dataset CWRU | 2, 3 | Diretório `data/raw` organizado e catalogado |
| **3** | Configuração do Ambiente Python e Vetores | 3, 4 | Ambiente virtual + script `data_loader.py` |
| **4** | Pré-processamento e Segmentação de Sinais | 4, 5 | Função de *sliding window* validada |
| **5** | Extração de Features (Tempo e FFT) | 5, 6 | Script `feature_extraction.py` + CSV gerado |
| **6** | Análise Exploratória e Seleção de Atributos | 6, 7 | Gráficos de correlação, boxplots e espectros |
| **7** | Classificador Base (Random Forest) | 7, 8 | Modelo baseline treinado e avaliado |
| **8** | Modelo Avançado (SVM) | 8, 9 | Modelo SVM treinado e avaliado |
| **9** | Ajuste de Hiperparâmetros e Otimização | 9, 10 | Resultados do `GridSearchCV` consolidados |
| **10** | Consolidação de Resultados e Gráficos | 10, 11 | Figuras finais e tabelas comparativas |
| **11** | Redação da Monografia Final | 11, 12 | Texto completo do TCC (Padrão UFSJ/ABNT) |
| **12** | Revisão Final e Preparação para Defesa | 12 | Slides de apresentação e repositório público |

---

## 7. Diretrizes de Atuação para o Agente de IA

Sempre que for solicitado a ajudar neste projeto, siga estas regras:
1. **Rigor Matemático e Físico:** Ao implementar funções de extração de atributos ou explicar conceitos para a monografia, relacione a matemática à física do problema (ex: explicar por que a *Curtose* aumenta quando uma esfera passa sobre uma fissura na pista).
2. **Qualidade de Código:** Escreva código Python limpo, tipado (*type hints*), documentado com *docstrings* e reproduzível (fixando `random_state=42` em todas as operações estocásticas).
3. **Prevenção de Vazamento de Dados (*Data Leakage*):** Sempre alerte e garanta que a separação entre treino e teste ocorra de maneira íntegra, especialmente ao usar janelas deslizantes com sobreposição.
4. **Foco Acadêmico (Padrão TCC):** Gráficos gerados via `matplotlib`/`seaborn` devem ter rótulos em português, unidades de medida claras nos eixos (ex: $\text{Frequência [Hz]}$, $\text{Aceleração [m/s}^2\text{ ou } g\text{]}$), fontes legíveis e formatação adequada para inserção direta na monografia.