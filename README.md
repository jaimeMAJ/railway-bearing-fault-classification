# Railway Bearing Fault Classification

Classificação de falhas em rolamentos ferroviários via Machine Learning, usando processamento de sinais de vibração (RMS, Curtose, Fator de Crista) e o dataset CWRU como benchmark. Trabalho de Conclusão de Curso — Engenharia Mecatrônica, UFSJ.

## Sobre o projeto

Este projeto implementa um pipeline de classificação de falhas em rolamentos a partir de sinais de vibração, cobrindo desde a ingestão de dados brutos até a avaliação de modelos de classificação. Como benchmark inicial, utiliza o dataset público [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter), permitindo validar a metodologia antes de aplicá-la a dados reais de rodeiros ferroviários.

**Classes de falha classificadas:**
- `Normal` — rolamento saudável (baseline)
- `IR` — falha na pista interna (Inner Race)
- `B` — falha em elemento rolante (Ball)
- `OR` — falha na pista externa, posição @6:00 (Outer Race)

Cada classe possui 3 níveis de severidade (0.007", 0.014", 0.021") e 4 condições de carga do motor (0 a 3 HP).

## Pipeline

1. **Download** (`download_cwru.py`) — baixa automaticamente os arquivos `.mat` (12 kHz, drive end) do site oficial da CWRU, com retry automático em caso de falha de conexão
2. **Ingestão e segmentação** (`src/preprocessing.py`) — lê os arquivos `.mat`, extrai o rótulo a partir do nome do arquivo e segmenta os sinais contínuos em janelas de tamanho fixo (não sobrepostas)
3. **Extração de features** (`src/features.py`) — calcula RMS, Curtose e Fator de Crista para cada janela
4. **Modelagem** (`notebooks/`) — divisão treino/teste por arquivo (evitando vazamento de dados entre janelas do mesmo ensaio), treino e avaliação de classificadores (Random Forest, SVM)

## Estrutura do repositório

```
notebook_python/
├── data/
│   ├── raw/          # arquivos .mat originais (não versionado, ver .gitignore)
│   └── processed/    # dados processados/features extraídas
├── notebooks/        # exploração e experimentos
├── src/
│   ├── preprocessing.py   # leitura, rótulos e segmentação
│   ├── features.py        # extração de features estatísticas
│   └── models.py          # (em desenvolvimento)
├── results/           # gráficos, métricas, modelos salvos
├── download_cwru.py   # script de download do dataset CWRU
├── requirements.txt
└── README.md
```

## Como reproduzir

```powershell
# 1. Clonar o repositório
git clone https://github.com/SEU-USUARIO/railway-bearing-fault-classification.git
cd railway-bearing-fault-classification

# 2. Criar e ativar o ambiente virtual
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Baixar o dataset CWRU
python download_cwru.py

# 5. Abrir os notebooks no VS Code e selecionar o kernel do venv
```

## Resultados preliminares

Primeiro modelo (Random Forest, sem ajuste de hiperparâmetros), usando RMS, Curtose e Fator de Crista como features, com divisão treino/teste estratificada por arquivo (70/30):

| Classe | Precision | Recall | F1-score |
|---|---|---|---|
| B (elemento rolante) | 0.97 | 0.81 | 0.89 |
| IR (pista interna) | 0.92 | 0.98 | 0.95 |
| Normal | 0.98 | 1.00 | 0.99 |
| OR (pista externa) | 0.95 | 0.99 | 0.97 |

**Acurácia geral: 95.49%**

A classe `B` apresenta o maior desafio de classificação, com confusões principalmente com `IR` e `OR` — resultado fisicamente coerente, já que falhas em elemento rolante tendem a gerar impactos mecânicos mais sutis que falhas de pista.

## Próximos passos

- [ ] Testar classificador SVM e comparar com Random Forest
- [ ] Investigar a origem das confusões da classe B por severidade
- [ ] Adicionar features no domínio da frequência (FFT)
- [ ] Ajuste de hiperparâmetros (Grid Search) e validação cruzada
- [ ] Aplicação da metodologia a dados reais de rodeiros ferroviários

## Autor

Jaime Junior — Engenharia Mecatrônica, UFSJ
Orientadora: Mariana Geny Moreira Vale Cardoso