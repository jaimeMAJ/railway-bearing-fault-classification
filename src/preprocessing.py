"""
src/preprocessing.py

Funções para ingestão dos arquivos .mat do CWRU Bearing Dataset, segmentação
por janelas (sliding window) e rastreabilidade temporal (data lineage).

Conforme as diretrizes da Seção 4 (Etapa 1) do plano de TCC.
"""

import os
import re
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.config import TAXA_AMOSTRAGEM, TAMANHO_JANELA_PADRAO


def _extrair_id_chave(chave: str) -> int:
    """Extrai o número identificador do sinal a partir do nome da chave (ex: 'X097_DE_time' -> 97)."""
    match = re.search(r"X(\d+)", chave)
    if match:
        return int(match.group(1))
    return 0


def carregar_sinal(caminho_arquivo: str) -> np.ndarray:
    """
    Lê um arquivo .mat do CWRU e retorna o sinal de vibração do Drive End (DE)
    como um vetor 1D achatado (N,).

    Trata casos de múltiplos sinais DE_time (como no Normal_2), selecionando
    o sinal correspondente ao maior ID do ensaio.
    """
    dados = loadmat(caminho_arquivo)
    chaves_de = [c for c in dados.keys() if "DE_time" in c]

    if not chaves_de:
        raise ValueError(f"Nenhuma coluna DE_time encontrada em {caminho_arquivo}")

    if len(chaves_de) > 1:
        chave_de = max(chaves_de, key=_extrair_id_chave)
    else:
        chave_de = chaves_de[0]

    sinal = dados[chave_de]
    return np.asarray(sinal).flatten()


def carregar_rpm(caminho_arquivo: str) -> Optional[float]:
    """
    Lê a velocidade de rotação (RPM) registrada no arquivo .mat.
    Retorna None se a chave de RPM não estiver presente.
    """
    dados = loadmat(caminho_arquivo)
    for chave in dados.keys():
        if "RPM" in chave:
            val = dados[chave].flatten()
            if len(val) > 0:
                return float(val[0])
    return None


def _separar_letras_numeros(texto: str) -> Tuple[str, str]:
    """Separa texto misto (ex: 'IR021') em ('IR', '021')."""
    letras = "".join(c for c in texto if c.isalpha())
    numeros = "".join(c for c in texto if c.isdigit())
    return letras, numeros


def extrair_rotulo(nome_arquivo: str) -> dict:
    """
    Extrai classe (4 classes), severidade, posição, carga e rótulo composto (10 classes)
    a partir do nome do arquivo CWRU.

    Exemplos:
        'Normal_0.mat'   -> classe='Normal', severidade=None, classe_10='Normal'
        'IR021_0.mat'    -> classe='IR',     severidade='021', classe_10='IR_021'
        'B007_3.mat'     -> classe='B',      severidade='007', classe_10='B_007'
        'OR014@6_2.mat'  -> classe='OR',     severidade='014', classe_10='OR_014'
    """
    nome = nome_arquivo.replace(".mat", "")
    partes = nome.split("_")
    carga = int(partes[-1])
    codigo = partes[0]

    if codigo == "Normal":
        return {
            "classe": "Normal",
            "severidade": None,
            "posicao": None,
            "carga": carga,
            "classe_10": "Normal",
        }

    posicao = None
    if "@" in codigo:
        codigo, posicao = codigo.split("@")

    letras, numeros = _separar_letras_numeros(codigo)
    classe_10 = f"{letras}_{numeros}" if numeros else letras

    return {
        "classe": letras,
        "severidade": numeros,
        "posicao": posicao,
        "carga": carga,
        "classe_10": classe_10,
    }


def preencher_rpm_ausente(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preenche a RPM ausente com a mediana da RPM dos arquivos de mesma carga do motor.
    Garante integridade referencial dos dados antes da extração de features espectrais.
    """
    df = df.copy()
    mediana_por_carga = df.groupby("carga")["rpm"].median()
    ausentes = df["rpm"].isna()

    if ausentes.any():
        df.loc[ausentes, "rpm"] = df.loc[ausentes, "carga"].map(mediana_por_carga)

    return df


def montar_dataframe_bruto(pasta_dados: str) -> pd.DataFrame:
    """
    Varre a pasta de dados brutos (.mat), carrega cada sinal e extrai metadados,
    retornando um DataFrame com granularidade de 1 linha por arquivo (ensaio).
    """
    registros = []
    arquivos = sorted(f for f in os.listdir(pasta_dados) if f.endswith(".mat"))

    for nome_arquivo in arquivos:
        caminho_completo = os.path.join(pasta_dados, nome_arquivo)
        sinal = carregar_sinal(caminho_completo)
        rpm = carregar_rpm(caminho_completo)
        rotulo = extrair_rotulo(nome_arquivo)

        registros.append({
            "arquivo": nome_arquivo,
            "classe": rotulo["classe"],
            "severidade": rotulo["severidade"],
            "posicao": rotulo["posicao"],
            "carga": rotulo["carga"],
            "classe_10": rotulo["classe_10"],
            "rpm": rpm,
            "sinal": sinal,
            "n_amostras": len(sinal),
            "duracao_s": len(sinal) / TAXA_AMOSTRAGEM,
        })

    return pd.DataFrame(registros)


def segmentar_sinal(
    sinal: np.ndarray,
    tamanho_janela: int = TAMANHO_JANELA_PADRAO,
    overlap: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segmenta um sinal 1D em janelas de tamanho fixo com overlap parametrizável.

    Retorna:
        janelas: matriz de shape (n_janelas, tamanho_janela)
        start_indices: vetor com o índice inicial de cada janela no sinal original
        end_indices: vetor com o índice final de cada janela no sinal original
    """
    if overlap < 0.0 or overlap >= 1.0:
        raise ValueError("O overlap deve estar no intervalo [0.0, 1.0).")

    passo = int(tamanho_janela * (1.0 - overlap))
    n_amostras = len(sinal)

    if n_amostras < tamanho_janela:
        return np.empty((0, tamanho_janela)), np.empty(0, dtype=int), np.empty(0, dtype=int)

    starts = list(range(0, n_amostras - tamanho_janela + 1, passo))
    janelas = np.array([sinal[i : i + tamanho_janela] for i in starts])
    start_indices = np.array(starts, dtype=int)
    end_indices = start_indices + tamanho_janela

    return janelas, start_indices, end_indices


def montar_dataframe_janelas(
    df: pd.DataFrame,
    tamanho_janela: int = TAMANHO_JANELA_PADRAO,
    overlap: float = 0.0,
) -> pd.DataFrame:
    """
    A partir do DataFrame de sinais brutos (1 linha por arquivo), gera o DataFrame
    de janelas (1 linha por janela), preservando rastreabilidade completa (data lineage):
    índice da janela, pontos de início/fim e rótulos herdados.
    """
    registros = []

    for _, linha in df.iterrows():
        janelas, starts, ends = segmentar_sinal(
            linha["sinal"],
            tamanho_janela=tamanho_janela,
            overlap=overlap,
        )

        for idx, (janela, start_idx, end_idx) in enumerate(zip(janelas, starts, ends)):
            registros.append({
                "arquivo": linha["arquivo"],
                "janela_idx": idx,
                "sample_start": start_idx,
                "sample_end": end_idx,
                "classe": linha["classe"],
                "severidade": linha["severidade"],
                "posicao": linha["posicao"],
                "carga": linha["carga"],
                "classe_10": linha["classe_10"],
                "rpm": linha["rpm"],
                "janela": janela,
            })

    return pd.DataFrame(registros)


if __name__ == "__main__":
    from src.config import RAW_DATA_DIR
    df_raw = montar_dataframe_bruto(str(RAW_DATA_DIR))
    df_raw = preencher_rpm_ausente(df_raw)
    print("Arquivos carregados:", len(df_raw))
    print(df_raw["classe"].value_counts())
