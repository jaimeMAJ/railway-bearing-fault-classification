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
from scipy.signal import hilbert

# Geometria do rolamento SKF 6205-2RS JEM (drive end) - CWRU Bearing Data Center
# Fonte: https://engineering.case.edu/bearingdatacenter/bearing-information
# Geometria detalhada: https://arxiv.org/pdf/2407.17236 (Tabela 1)
DIAMETRO_ESFERA = 0.3126   # polegadas (d)
DIAMETRO_PASSO = 1.537     # polegadas (D)
NUM_ESFERAS = 9            # n
ANGULO_CONTATO = 0         # graus (phi) - cos(0°) = 1, rolamento radial


def calcular_frequencias_falha(rpm: float) -> dict:
    """
    Calcula as frequências teóricas de falha (BPFO, BPFI, BSF) em Hz,
    a partir da RPM do eixo e da geometria do rolamento SKF 6205.

    Fórmulas (com phi = 0°, cos(phi) = 1):
    https://www.tryharmony.ai/bearing-defect-frequencies
    https://vibromera.eu/glossary/bpfo/
    """
    fr = rpm / 60  # RPM -> Hz
    razao = DIAMETRO_ESFERA / DIAMETRO_PASSO

    bpfo = (NUM_ESFERAS * fr / 2) * (1 - razao)
    bpfi = (NUM_ESFERAS * fr / 2) * (1 + razao)
    bsf = (DIAMETRO_PASSO * fr / (2 * DIAMETRO_ESFERA)) * (1 - razao ** 2)

    return {"bpfo": bpfo, "bpfi": bpfi, "bsf": bsf}


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

def calcular_envelope(janela: np.ndarray) -> np.ndarray:
    """
    Envelope de amplitude via Transformada de Hilbert. Revela a modulação
    de baixa frequência (impacto periódico do defeito) escondida sob a
    ressonância estrutural de alta frequência.
    """
    return np.abs(hilbert(janela))


def calcular_fft(janela: np.ndarray, taxa_amostragem: int = 12000) -> tuple:
    """
    Espectro de magnitude (FFT) de um sinal real. Retorna (frequencias, magnitudes),
    mantendo só a metade positiva do espectro (a outra metade é espelhada).
    """
    n = len(janela)
    magnitudes = np.abs(np.fft.fft(janela))
    frequencias = np.fft.fftfreq(n, d=1 / taxa_amostragem)
    metade = n // 2
    return frequencias[:metade], magnitudes[:metade]


def calcular_energia_frequencia(freqs: np.ndarray, mags: np.ndarray,
                                freq_alvo: float, tolerancia: float = 5.0) -> float:
    """
    Magnitude máxima do espectro na faixa freq_alvo +/- tolerancia (Hz).
    Usa uma faixa, e não um ponto exato, porque a FFT é discreta e a RPM
    real varia levemente em relação à teórica.
    """
    mascara = (freqs >= freq_alvo - tolerancia) & (freqs <= freq_alvo + tolerancia)
    if not mascara.any():
        return 0.0
    return mags[mascara].max()


def calcular_features_frequencia(janela: np.ndarray, rpm: float,
                                 taxa_amostragem: int = 12000) -> dict:
    """
    Features de energia espectral perto de BPFO, BPFI e BSF, calculadas
    sobre o espectro do envelope (Análise de Envelope).
    """
    envelope = calcular_envelope(janela)
    freqs, mags = calcular_fft(envelope, taxa_amostragem)
    teoricas = calcular_frequencias_falha(rpm)

    return {
        "energia_bpfo": calcular_energia_frequencia(freqs, mags, teoricas["bpfo"]),
        "energia_bpfi": calcular_energia_frequencia(freqs, mags, teoricas["bpfi"]),
        "energia_bsf": calcular_energia_frequencia(freqs, mags, teoricas["bsf"]),
    }

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
            "fator_crista": calcular_fator_crista(janela),
            **calcular_features_frequencia(janela, linha["rpm"]),
        })

    return pd.DataFrame(registros)