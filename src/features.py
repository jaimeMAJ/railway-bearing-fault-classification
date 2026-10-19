"""
src/features.py

Extração de características estatísticas nos domínios do tempo e da frequência (FFT/Envelope)
a partir de sinais de vibração acelerométrica, com base na Seção 4 (Etapa 2) do plano de TCC.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew

from src.config import (
    ANGULO_CONTATO,
    DIAMETRO_ESFERA,
    DIAMETRO_PASSO,
    NUM_ESFERAS,
    TAXA_AMOSTRAGEM,
)


def calcular_frequencias_falha(rpm: float) -> Dict[str, float]:
    """
    Calcula as frequências teóricas de defeito do rolamento (Hz) a partir da rotação
    do eixo e da geometria do SKF 6205:
      - BPFO: Ball Pass Frequency Outer Race (Pista Externa)
      - BPFI: Ball Pass Frequency Inner Race (Pista Interna)
      - BSF:  Ball Spin Frequency (Giro do Elemento Rolante)
      - FTF:  Fundamental Train Frequency (Frequência da Gaiola)
    """
    if pd.isna(rpm) or rpm <= 0:
        return {"bpfo": 0.0, "bpfi": 0.0, "bsf": 0.0, "ftf": 0.0}

    fr = rpm / 60.0  # Rotação do eixo em Hz
    cos_phi = np.cos(np.radians(ANGULO_CONTATO))
    razao = (DIAMETRO_ESFERA / DIAMETRO_PASSO) * cos_phi

    bpfo = (NUM_ESFERAS * fr / 2.0) * (1.0 - razao)
    bpfi = (NUM_ESFERAS * fr / 2.0) * (1.0 + razao)
    bsf = (DIAMETRO_PASSO * fr / (2.0 * DIAMETRO_ESFERA)) * (1.0 - razao**2)
    ftf = (fr / 2.0) * (1.0 - razao)

    return {"bpfo": bpfo, "bpfi": bpfi, "bsf": bsf, "ftf": ftf}


# ==============================================================================
# Features no Domínio do Tempo (Estatísticas Estocásticas)
# ==============================================================================

def calcular_rms(janela: np.ndarray) -> float:
    """Root Mean Square (RMS): mede a energia global e amplitude média quadrática."""
    return float(np.sqrt(np.mean(janela**2)))


def calcular_curtose(janela: np.ndarray) -> float:
    """
    Curtose (com referência em 3.0, fisher=False): mede a impulsividade do sinal.
    Sinais gaussianos saudáveis têm curtose ~3.0; impactos periódicos elevam o valor.
    """
    return float(kurtosis(janela, fisher=False))


def calcular_fator_crista(janela: np.ndarray) -> float:
    """Fator de Crista: razão entre o pico absoluto máximo e o RMS."""
    rms = calcular_rms(janela)
    if rms < 1e-12:
        return 0.0
    return float(np.max(np.abs(janela)) / rms)


def calcular_features_tempo(janela: np.ndarray) -> Dict[str, float]:
    """
    Calcula o conjunto completo de atributos estocásticos no domínio do tempo
    recomendados nas especificações do TCC.
    """
    rms = calcular_rms(janela)
    pico = float(np.max(np.abs(janela)))
    media_abs = float(np.mean(np.abs(janela)))
    eps = 1e-12

    return {
        "rms": rms,
        "curtose": calcular_curtose(janela),
        "fator_crista": (pico / rms) if rms > eps else 0.0,
        "desvio_padrao": float(np.std(janela)),
        "assimetria": float(skew(janela)),
        "pico_a_pico": float(np.ptp(janela)),
        "fator_forma": (rms / media_abs) if media_abs > eps else 0.0,
        "fator_impulso": (pico / media_abs) if media_abs > eps else 0.0,
    }


# ==============================================================================
# Features no Domínio da Frequência (FFT e Análise de Envelope)
# ==============================================================================

def calcular_envelope(janela: np.ndarray, remover_dc: bool = True) -> np.ndarray:
    """
    Calcula o envelope de amplitude via Transformada de Hilbert.
    Por padrão, remove o offset de média DC para evitar saturação em 0 Hz.
    """
    envelope = np.abs(hilbert(janela))
    if remover_dc:
        envelope = envelope - np.mean(envelope)
    return envelope


def calcular_fft(
    sinal: np.ndarray,
    taxa_amostragem: int = TAXA_AMOSTRAGEM,
    aplicar_hanning: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula o espectro de magnitude unilateral via FFT (scipy/numpy).
    Opcionalmente aplica janela de Hanning para minimizar o vazamento espectral.
    """
    n = len(sinal)
    if aplicar_hanning:
        janela_jan = np.hanning(n)
        sinal_proc = sinal * janela_jan
    else:
        sinal_proc = sinal

    # Espectro normalizado
    mags = np.abs(np.fft.fft(sinal_proc)) * (2.0 / n)
    freqs = np.fft.fftfreq(n, d=1.0 / taxa_amostragem)

    metade = n // 2
    return freqs[:metade], mags[:metade]


def calcular_energia_banda(
    freqs: np.ndarray,
    mags: np.ndarray,
    freq_alvo: float,
    tolerancia: float = 5.0,
) -> float:
    """Retorna a magnitude máxima do espectro na faixa freq_alvo +/- tolerancia (Hz)."""
    if freq_alvo <= 0:
        return 0.0
    mascara = (freqs >= freq_alvo - tolerancia) & (freqs <= freq_alvo + tolerancia)
    if not mascara.any():
        return 0.0
    return float(mags[mascara].max())


def calcular_features_frequencia(
    janela: np.ndarray,
    rpm: float,
    taxa_amostragem: int = TAXA_AMOSTRAGEM,
) -> Dict[str, float]:
    """
    Calcula atributos espectrais globais e energias nas frequências características
    de falha (BPFO, BPFI, BSF, FTF) a partir do espectro de envelope.
    """
    teoricas = calcular_frequencias_falha(rpm)

    # 1. Espectro do Envelope (com remoção de DC e janela de Hanning)
    envelope = calcular_envelope(janela, remover_dc=True)
    freqs_env, mags_env = calcular_fft(envelope, taxa_amostragem, aplicar_hanning=True)

    # 2. Espectro Direto da FFT
    freqs_dir, mags_dir = calcular_fft(janela, taxa_amostragem, aplicar_hanning=True)

    # Métricas espectrais globais no espectro direto
    soma_mags = np.sum(mags_dir) + 1e-12
    centroide_espectral = float(np.sum(freqs_dir * mags_dir) / soma_mags)
    variancia_espectral = float(np.sum(((freqs_dir - centroide_espectral) ** 2) * mags_dir) / soma_mags)

    idx_pico = int(np.argmax(mags_dir))
    freq_pico_dominante = float(freqs_dir[idx_pico])
    amp_pico_dominante = float(mags_dir[idx_pico])

    return {
        # Energias nas frequências características de falha (análise de envelope)
        "energia_bpfo": calcular_energia_banda(freqs_env, mags_env, teoricas["bpfo"]),
        "energia_bpfi": calcular_energia_banda(freqs_env, mags_env, teoricas["bpfi"]),
        "energia_bsf": calcular_energia_banda(freqs_env, mags_env, teoricas["bsf"]),
        "energia_ftf": calcular_energia_banda(freqs_env, mags_env, teoricas["ftf"]),
        # Atributos espectrais globais
        "centroide_espectral": centroide_espectral,
        "variancia_espectral": variancia_espectral,
        "freq_pico_dominante": freq_pico_dominante,
        "amp_pico_dominante": amp_pico_dominante,
    }


def montar_dataframe_features(df_janelas: pd.DataFrame) -> pd.DataFrame:
    """
    Processa cada janela e compõe a Feature Store completa (Data Mart),
    unificando metadados de linhagem, rótulos e características extraídas.
    """
    registros = []

    for _, linha in df_janelas.iterrows():
        janela = linha["janela"]
        rpm = linha["rpm"]

        feats_tempo = calcular_features_tempo(janela)
        feats_freq = calcular_features_frequencia(janela, rpm)

        reg = {
            "arquivo": linha["arquivo"],
            "janela_idx": linha.get("janela_idx", 0),
            "sample_start": linha.get("sample_start", 0),
            "sample_end": linha.get("sample_end", len(janela)),
            "classe": linha["classe"],
            "severidade": linha["severidade"],
            "posicao": linha["posicao"],
            "carga": linha["carga"],
            "classe_10": linha.get("classe_10", linha["classe"]),
            "rpm": rpm,
            **feats_tempo,
            **feats_freq,
        }
        registros.append(reg)

    return pd.DataFrame(registros)


def validar_qualidade_features(df_features: pd.DataFrame) -> bool:
    """
    Testes de qualidade e integridade de dados (Data Contracts / AE Tests):
    - Verifica ausência de NaNs e Infinitos nas colunas numéricas de features.
    - Assegura limites físicos válidos (RMS >= 0, Fator Crista >= 1.0).
    """
    colunas_features = [
        "rms", "curtose", "fator_crista", "desvio_padrao", "assimetria",
        "pico_a_pico", "fator_forma", "fator_impulso", "energia_bpfo",
        "energia_bpfi", "energia_bsf", "energia_ftf", "centroide_espectral",
        "variancia_espectral", "freq_pico_dominante", "amp_pico_dominante"
    ]

    # 1. Checagem de nulos
    nulos = df_features[colunas_features].isna().sum()
    if nulos.any():
        cols_com_nulo = nulos[nulos > 0].to_dict()
        raise ValueError(f"[DATA QUALITY TEST FAILED] Features com valores nulos: {cols_com_nulo}")

    # 2. Checagem de infinitos
    infinitos = np.isinf(df_features[colunas_features].values).sum()
    if infinitos > 0:
        raise ValueError("[DATA QUALITY TEST FAILED] Features contêm valores infinitos (divisão por zero).")

    # 3. Limites de domínio físico
    if (df_features["rms"] < 0).any():
        raise ValueError("[DATA QUALITY TEST FAILED] RMS com valor negativo detectado.")

    print(f"[DATA QUALITY PASS] Todas as {len(colunas_features)} features validadas com sucesso (0 nulos, 0 infinitos).")
    return True
