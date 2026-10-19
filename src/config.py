"""
src/config.py

Configurações centrais, parâmetros físicos do rolamento e constantes do projeto
conforme as especificações do TCC (Engenharia Mecatrônica - UFSJ).
"""

from pathlib import Path

# Caminhos padrão do projeto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Taxa de amostragem padrão (Dataset CWRU - Drive End)
TAXA_AMOSTRAGEM = 12000  # 12 kHz

# Tamanho padrão da janela de segmentação e overlap
TAMANHO_JANELA_PADRAO = 2048
OVERLAP_PADRAO = 0.0

# Semente pseudoaleatória para reprodutibilidade estocástica
RANDOM_STATE = 42

# Geometria do rolamento SKF 6205-2RS JEM (Drive End - CWRU)
# Fonte oficial CWRU: https://engineering.case.edu/bearingdatacenter/bearing-information
# Referência: arXiv:2407.17236 (Tabela 1)
DIAMETRO_ESFERA = 0.3126   # polegadas (d)
DIAMETRO_PASSO = 1.537     # polegadas (D)
NUM_ESFERAS = 9            # n (número de esferas)
ANGULO_CONTATO = 0.0       # graus (phi) - rolamento radial puro: cos(0°) = 1.0

# Mapeamento de rótulos amigáveis para gráficos acadêmicos
NOMES_CLASSES_PT = {
    "Normal": "Saudável (Baseline)",
    "IR": "Falha na Pista Interna (IR)",
    "B": "Falha em Elemento Rolante (B)",
    "OR": "Falha na Pista Externa (OR)",
}
