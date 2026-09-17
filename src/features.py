"""
src/features.py

Funções de extração de features estatísticas no domínio do tempo, a partir
das janelas geradas pela etapa de segmentação (ver preprocessing.py).

Features implementadas (conforme o plano do TCC):
- RMS (Root Mean Square): energia/amplitude média do sinal.
- Curtose (Kurtosis): "impulsividade" do sinal — sensível a picos raros e
  extremos, típicos do impacto de um defeito localizado no rolamento.
- Fator de Crista (Crest Factor): razão entre o pico e o RMS.
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis


def calcular_rms(janela: np.ndarray) -> float:
    """Root Mean Square: raiz da média dos quadrados dos pontos da janela."""
    return np.sqrt(np.mean(janela ** 2))


def calcular_curtose(janela: np.ndarray) -> float:
    """
    Curtose com referência em 3.0 (fisher=False), convenção comum na
    literatura de manutenção preditiva. Um valor próximo de 3 indica sinal
    aproximadamente uniforme; valores bem acima de 3 indicam picos raros
    e extremos (comportamento típico de impacto por defeito).
    """
    return kurtosis(janela, fisher=False)


def calcular_fator_crista(janela: np.ndarray) -> float:
    """Razão entre o valor de pico absoluto e o RMS da janela."""
    pico = np.max(np.abs(janela))
    rms = calcular_rms(janela)
    return pico / rms


def montar_dataframe_features(df_janelas: pd.DataFrame) -> pd.DataFrame:
    """
    A partir do DataFrame de janelas (1 linha por janela, coluna 'janela'
    com o array bruto), calcula RMS, Curtose e Fator de Crista de cada
    janela e retorna um DataFrame só com os rótulos + as features
    (pronto para uso em modelos de Machine Learning).
    """
    registros = []

    for _, linha in df_janelas.iterrows():
        janela = linha["janela"]

        registros.append({
            "arquivo": linha["arquivo"],
            "classe": linha["classe"],
            "severidade": linha["severidade"],
            "posicao": linha["posicao"],
            "carga": linha["carga"],
            "rms": calcular_rms(janela),
            "curtose": calcular_curtose(janela),
            "fator_crista": calcular_fator_crista(janela),
        })

    return pd.DataFrame(registros)