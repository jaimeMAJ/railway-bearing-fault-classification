"""
src/preprocessing.py

Funções para leitura dos arquivos .mat do CWRU Bearing Dataset e
organização inicial em um DataFrame único (sinal bruto + rótulo por arquivo).

Este módulo cobre a etapa de "Ingestão" do pipeline descrito no plano do TCC.
A segmentação por janelas deslizantes e a extração de features (RMS, Curtose,
FFT etc.) ficam em módulos separados (features.py), para manter cada etapa
do pipeline independente e fácil de testar isoladamente.
"""

import os

import numpy as np
import pandas as pd
from scipy.io import loadmat


def carregar_sinal(caminho_arquivo: str) -> np.ndarray:
    """
    Lê um arquivo .mat do CWRU e retorna o sinal de vibração do Drive End (DE)
    como um vetor 1D (achatado).

    O nome da variável dentro do .mat muda de arquivo para arquivo
    (ex: 'X097_DE_time', 'X209_DE_time'), pois inclui o ID interno do arquivo.
    Por isso, em vez de tentar adivinhar o nome, procuramos dinamicamente
    a chave que contém 'DE_time'.
    """
    dados = loadmat(caminho_arquivo)

    chave_de = None
    for chave in dados.keys():
        if "DE_time" in chave:
            chave_de = chave
            break

    if chave_de is None:
        raise ValueError(f"Nenhuma coluna DE_time encontrada em {caminho_arquivo}")

    sinal = dados[chave_de]
    return sinal.flatten()  # (N, 1) -> (N,)


def _separar_letras_numeros(texto: str) -> tuple[str, str]:
    """Separa um texto misto (ex: 'IR021') em (letras, numeros) = ('IR', '021')."""
    letras = ""
    numeros = ""
    for caractere in texto:
        if caractere.isalpha():
            letras += caractere
        elif caractere.isdigit():
            numeros += caractere
    return letras, numeros


def extrair_rotulo(nome_arquivo: str) -> dict:
    """
    Extrai classe, severidade, posição (se houver) e carga a partir do
    nome do arquivo, seguindo o padrão de nomenclatura da CWRU.

    Exemplos:
        'Normal_0.mat'   -> classe='Normal', severidade=None, posicao=None, carga=0
        'IR021_0.mat'    -> classe='IR',     severidade='021', posicao=None, carga=0
        'B007_3.mat'     -> classe='B',      severidade='007', posicao=None, carga=3
        'OR014@6_2.mat'  -> classe='OR',     severidade='014', posicao='6',  carga=2
    """
    nome = nome_arquivo.replace(".mat", "")

    partes = nome.split("_")
    carga = int(partes[-1])
    codigo = partes[0]

    if codigo == "Normal":
        return {"classe": "Normal", "severidade": None, "posicao": None, "carga": carga}

    posicao = None
    if "@" in codigo:
        codigo, posicao = codigo.split("@")

    letras, numeros = _separar_letras_numeros(codigo)

    return {
        "classe": letras,
        "severidade": numeros,
        "posicao": posicao,
        "carga": carga,
    }


def montar_dataframe_bruto(pasta_dados: str) -> pd.DataFrame:
    """
    Varre a pasta de dados brutos (.mat), carrega cada sinal e extrai seu
    rótulo, retornando um DataFrame com uma linha por arquivo.

    Colunas: arquivo, classe, severidade, posicao, carga, sinal, n_amostras
    """
    registros = []

    arquivos = sorted(f for f in os.listdir(pasta_dados) if f.endswith(".mat"))

    for nome_arquivo in arquivos:
        caminho_completo = os.path.join(pasta_dados, nome_arquivo)

        sinal = carregar_sinal(caminho_completo)
        rotulo = extrair_rotulo(nome_arquivo)

        registros.append({
            "arquivo": nome_arquivo,
            "classe": rotulo["classe"],
            "severidade": rotulo["severidade"],
            "posicao": rotulo["posicao"],
            "carga": rotulo["carga"],
            "sinal": sinal,
            "n_amostras": len(sinal),
        })

    return pd.DataFrame(registros)


if __name__ == "__main__":
    # Execução direta (fora do notebook) para teste rápido via terminal:
    # python src/preprocessing.py
    df = montar_dataframe_bruto("data/raw")
    print(df.drop(columns="sinal"))  # não imprime o array inteiro, só as demais colunas
    print("\nDistribuição de classes:")
    print(df["classe"].value_counts())


def segmentar_sinal(sinal: np.ndarray, tamanho_janela: int) -> np.ndarray:
    """
    Divide um sinal 1D em janelas não sobrepostas de tamanho fixo.
    Descarta o resto que não forma uma janela completa.

    Retorna uma matriz de shape (n_janelas, tamanho_janela).
    """
    n_janelas = len(sinal) // tamanho_janela
    sinal_util = sinal[: n_janelas * tamanho_janela]  # corta o "resto" fora
    janelas = sinal_util.reshape(n_janelas, tamanho_janela)
    return janelas

def montar_dataframe_janelas(df: pd.DataFrame, tamanho_janela: int) -> pd.DataFrame:
    """
    A partir do DataFrame de sinais brutos (1 linha por arquivo), gera um
    novo DataFrame com 1 linha por janela, repetindo o rótulo do arquivo
    de origem em cada janela.
    """
    registros = []

    for _, linha in df.iterrows():
        janelas = segmentar_sinal(linha["sinal"], tamanho_janela)

        for janela in janelas:
            registros.append({
                "arquivo": linha["arquivo"],
                "classe": linha["classe"],
                "severidade": linha["severidade"],
                "posicao": linha["posicao"],
                "carga": linha["carga"],
                "janela": janela,
            })

    return pd.DataFrame(registros)