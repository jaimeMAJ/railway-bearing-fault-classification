# Plano de Execução: Estrutura e Redação da Monografia em LaTeX (abnTeX2)

**Instrução ao Agente:** Utilize a skill de redação acadêmica já configurada no projeto para executar as etapas abaixo em sequência. Analise o código Python existente no repositório (`src/`, notebooks ou scripts de modelagem) para extrair os parâmetros reais utilizados (tamanho de janela, *overlap*, features extraídas, grade de hiperparâmetros e métricas de avaliação) e integrá-los diretamente ao texto e às tabelas LaTeX.

---

## Etapa 1: Criação da Árvore de Diretórios

Garanta que a seguinte estrutura de diretórios exista na raiz do projeto (crie as pastas caso ainda não existam):

```text
├── monografia/
│   ├── main.tex
│   ├── referencias.bib
│   ├── Makefile
│   ├── pre_textuais/
│   │   └── resumo.tex
│   └── capitulos/
│       ├── 01_introducao.tex
│       ├── 02_revisao.tex
│       ├── 03_metodologia.tex
│       ├── 04_resultados.tex
│       └── 05_conclusao.tex
└── results/
    ├── figures/
    └── tables/

```

---

## Etapa 2: Configuração dos Arquivos Base do LaTeX (`abnTeX2`)

### 2.1. Arquivo Principal (`monografia/main.tex`)

Crie o arquivo `monografia/main.tex` utilizando a classe `abntex2` configurada com os seguintes requisitos:

* **Classe:** `\documentclass[12pt,openright,oneside,a4paper,english,brazil]{abntex2}`
* **Pacotes Obrigatórios:**
* Codificação e fontes: `inputenc` (`utf8`), `fontenc` (`T1`), `lmodern`, `microtype`.
* Matemática e Unidades: `amsmath`, `amsfonts`, `amssymb`, `siunitx` (configurado com `output-decimal-marker = {,}`).
* Elementos visuais e tabelas: `graphicx`, `booktabs`, `multirow`, `float`, `subcaption`.
* Citações ABNT: `\usepackage[alf,abnt-emphasize=bf,abnt-etal-list=0]{abntex2cite}`.


* **Caminho de Imagens:** `\graphicspath{{../results/figures/}}`
* **Metadados Institucionais:**
* `\instituicao{Universidade Federal de São João del-Rei \par Curso de Engenharia Mecatrônica}`
* `\autor{Jaime Malafaia de Arêdes Junior}`
* `\titulo{Desenvolvimento de Modelo Preditivo Baseado em Inteligência Artificial para Classificação de Falhas em Rolamentos de Rodeiros Ferroviários}`
* `\orientadora{Profa. Mariana Geny Moreira Vale Cardoso}`
* `\local{São João del-Rei -- MG}`
* `\data{2026}`
* `\tipotrabalho{Trabalho de Conclusão de Curso (Graduação)}`
* `\preambulo{Trabalho de Conclusão de Curso apresentado à Coordenadoria do Curso de Engenharia Mecatrônica da Universidade Federal de São João del-Rei como requisito parcial para obtenção do título de Bacharel em Engenharia Mecatrônica.}`


* **Estrutura do Documento (`\begin{document}`):**
* `\imprimircapa` e `\imprimirfolhaderosto*`
* `\include{pre_textuais/resumo}`
* `\pdfbookmark[0]{\listfigurename}{lof} \listoffigures* \cleardoublepage`
* `\pdfbookmark[0]{\listtablename}{lot} \listoftables* \cleardoublepage`
* `\pdfbookmark[0]{\contentsname}{toc} \tableofcontents* \cleardoublepage`
* `\textual`
* `\include{capitulos/01_introducao}`
* `\include{capitulos/02_revisao}`
* `\include{capitulos/03_metodologia}`
* `\include{capitulos/04_resultados}`
* `\include{capitulos/05_conclusao}`
* `\postextual`
* `\bibliography{referencias}`



### 2.2. Resumo e Abstract (`monografia/pre_textuais/resumo.tex`)

Crie os ambientes `\begin{resumo}` e `\begin{resumo}[Abstract]` (com `\begin{otherlanguage*}{english}`) contendo o resumo e o abstract do plano de trabalho, incluindo as palavras-chave: *Manutenção Preditiva. Rolamentos Ferroviários. Processamento de Sinais. Machine Learning. CWRU Dataset.*

### 2.3. Base Bibliográfica (`monografia/referencias.bib`)

Crie o arquivo `monografia/referencias.bib` com entradas BibTeX completas e válidas para fundamentar as citações do texto, incluindo obrigatoriamente:

* `smith2015rolling`: Smith, W. A.; Randall, R. B. *Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study*. Mechanical Systems and Signal Processing, v. 64-65, p. 100-131, 2015.
* `randall2011rolling`: Randall, R. B.; Antoni, J. *Rolling element bearing diagnostics—A tutorial*. Mechanical Systems and Signal Processing, v. 25, n. 2, p. 485-520, 2011.
* `breiman2001random`: Breiman, L. *Random Forests*. Machine Learning, v. 45, n. 1, p. 5-32, 2001.
* `cortes1995support`: Cortes, C.; Vapnik, V. *Support-vector networks*. Machine Learning, v. 20, n. 3, p. 273-297, 1995.
* `oppenheim2010discrete`: Oppenheim, A. V.; Schafer, R. W. *Discrete-Time Signal Processing*. 3rd ed. Prentice Hall, 2010.
* `pedregosa2011scikit`: Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, v. 12, p. 2825-2830, 2011.

### 2.4. Automação de Build (`monografia/Makefile`)

Crie um `Makefile` dentro de `monografia/` com regras para compilar o documento (`pdflatex` + `bibtex` ou `latexmk -pdf main.tex`) e uma regra `clean` para limpar arquivos auxiliares (`.aux`, `.log`, `.bbl`, `.blg`, `.toc`, `.lof`, `.lot`).

---

## Etapa 3: Integração Código Python -> Artefatos do LaTeX

Inspecione os scripts Python existentes no repositório e crie/execute um script auxiliar (ex.: `src/export_latex_artifacts.py`) que gere e salve automaticamente:

1. **Tabelas em LaTeX (`results/tables/`):**
* `tab_dataset_classes.tex`: Mapeamento das classes do dataset CWRU utilizadas no código (Condição Normal, Pista Interna [IR], Pista Externa [OR], Elementos Rolantes [Ball], diâmetros de falha de $\SI{0.007}{in}$, $\SI{0.014}{in}$ e $\SI{0.021}{in}$, e número de janelas extraídas por classe).
* `tab_hiperparametros.tex`: Espaço de busca do `GridSearchCV` e os melhores hiperparâmetros encontrados para o Random Forest e para o SVM.
* `tab_resultados_modelos.tex`: Tabela comparativa no padrão IBGE/ABNT (`\toprule`, `\midrule`, `\bottomrule`, sem bordas verticais) com Acurácia, Precisão, Recall e F1-Score de cada modelo no conjunto de teste.


2. **Figuras em Alta Resolução (`results/figures/` em formato `.pdf` ou `.png` a 300 DPI):**
* `fig_sinais_tempo_fft.pdf`: Exemplo comparativo de um sinal no domínio do tempo e seu espectro FFT para uma condição saudável vs. com falha.
* `fig_matriz_confusao_rf.pdf` e `fig_matriz_confusao_svm.pdf`: Matrizes de confusão geradas pelos modelos treinados.
* `fig_feature_importance.pdf`: Importância dos atributos extraídos (no tempo e na frequência) segundo o Random Forest.



---

## Etapa 4: Redação Completa dos Capítulos (`monografia/capitulos/`)

Redija o conteúdo acadêmico completo e aprofundado de cada capítulo (iniciando diretamente em `\chapter{...}`, sem preâmbulo repetido):

### 4.1. `01_introducao.tex`

* **1.1 Contextualização e Definição do Problema:** Papel estratégico do setor ferroviário, solicitação dinâmica nas caixas de mancal (*axleboxes*) dos rodeiros ferroviários e riscos catastróficos de travamento de eixo e descarrilamento.
* **1.2 Justificativa:** Limitações da manutenção preventiva baseada em tempo/quilometragem e de sensores térmicos de via (*hotbox detectors*, que detectam falhas apenas em estágios terminais) frente à análise preditiva de vibração (*Condition-Based Maintenance* -- CBM). Viabilidade científica do benchmark da CWRU ($\SI{12}{\kilo\hertz}$) e integração interdisciplinar da Engenharia Mecatrônica (Mecânica, Eletrônica/Instrumentação e Computação).
* **1.3 Objetivos (Geral e Específicos):** Formalização dos objetivos do projeto.
* **1.4 Organização do Trabalho:** Descrição sucinta dos capítulos da monografia.

### 4.2. `02_revisao.tex`

* **2.1 Dinâmica e Modos de Falha em Rolamentos de Elementos Rolantes:** Geometria dos rolamentos e equacionamento das frequências características de defeito: *Ball Pass Frequency Outer* (BPFO), *Ball Pass Frequency Inner* (BPFI), *Ball Spin Frequency* (BSF) e *Fundamental Train Frequency* (FTF), com todas as variáveis definidas.
* **2.2 Processamento Digital de Sinais Aplicado à Análise de Vibrações:**
* Formulação matemática dos descritores estatísticos no domínio do tempo: Valor Eficaz (*Root Mean Square* -- RMS), Curtose (*Kurtosis*), Fator de Crista (*Crest Factor*), Desvio Padrão, Assimetria (*Skewness*) e Pico a Pico (conforme implementados no código Python).
* Transformada Discreta de Fourier (DFT) e Transformada Rápida de Fourier (FFT) para análise espectral.


* **2.3 Aprendizado de Máquina na Classificação de Falhas:**
* *Random Forest*: Conceito de *bagging*, árvores de decisão e critério de impureza de Gini/Entropia `\cite{breiman2001random}`.
* *Support Vector Machines* (SVM): Hiperplanos de margem máxima, variáveis de folga, formulação dual e funções de *kernel* (RBF, linear, polinomial) `\cite{cortes1995support}`.
* Métricas de avaliação multiclasse: Matriz de Confusão, Acurácia, Precisão, Sensibilidade (*Recall*) e *F1-Score*.



### 4.3. `03_metodologia.tex`

* Descreva detalhadamente o pipeline implementado no repositório Python (`SciPy`, `NumPy`, `Pandas`, `Scikit-Learn`):
* **3.1 Base de Dados Experimental (CWRU Bearing Dataset):** Especificações da bancada de testes da CWRU, aquisição a $\SI{12}{\kilo\hertz}$ no mancal *Drive End* (DE) e inclusão da Tabela `tab_dataset_classes.tex` via `\input{../results/tables/tab_dataset_classes.tex}`.
* **3.2 Pré-processamento e Segmentação por Janelas Deslizantes:** Explicação matemática da técnica de *sliding window* (tamanho de janela $N$ e passo de sobreposição adotados no código) para aumento de amostras e estacionariedade local.
* **3.3 Engenharia e Extração de Atributos:** Detalhamento exato do vetor de características (*feature vector*) extraído de cada janela no tempo e na frequência no código do projeto.
* **3.4 Treinamento, Validação Cruzada e Otimização de Hiperparâmetros:** Divisão estratificada treino/teste ($70\% / 30\%$), normalização (*StandardScaler*), validação cruzada $k$-fold e otimização via *Grid Search*, referenciando `tab_hiperparametros.tex`.



### 4.4. `04_resultados.tex`

* **4.1 Análise Exploratória dos Sinais e Atributos:** Discussão do comportamento dos sinais no tempo e espectro FFT (referenciando `Figura~\ref{fig:sinais_tempo_fft}`) e análise da relevância dos extratores (`Figura~\ref{fig:feature_importance}`).
* **4.2 Desempenho dos Classificadores (Random Forest vs. SVM):** Apresentação da tabela comparativa (`\input{../results/tables/tab_resultados_modelos.tex}`) e discussão quantitativa das métricas obtidas nos scripts.
* **4.3 Análise de Erros por Tipo e Severidade de Falha:** Discussão crítica das Matrizes de Confusão (`Figura~\ref{fig:matriz_confusao_rf}` e `Figura~\ref{fig:matriz_confusao_svm}`), interpretando fisicamente os resultados sob a ótica da mecânica de vibrações (ex.: estacionariedade da zona de carga em falhas na pista externa versus modulação de transferência nas falhas de elementos rolantes e pista interna nas severidades de $\SI{0.007}{in}$, $\SI{0.014}{in}$ e $\SI{0.021}{in}$).

### 4.5. `05_conclusao.tex`

* Síntese dos resultados alcançados frente aos objetivos propostos, validação da eficácia dos extratores de tempo/FFT combinados aos classificadores supervisionados para manutenção preditiva ferroviária e propostas concretas de trabalhos futuros (implementação do modelo otimizado em sistemas embarcados/microcontroladores de borda, testes com variação de carga e rotação e fusão sensorial).
