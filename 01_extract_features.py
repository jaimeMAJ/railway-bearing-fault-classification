## Importando bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing import (
    montar_dataframe_bruto, segmentar_sinal, montar_dataframe_janelas, carregar_rpm, preencher_rpm_ausente
)
from src.features import montar_dataframe_features, calcular_frequencias_falha
import os
from scipy.stats import kurtosis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

os.makedirs("results", exist_ok=True)

# Primeiros teste carregando o arquivo .mat

dados = loadmat("data/raw/Normal_0.mat")
print(dados.keys())

sinal = dados['X097_DE_time']
print("Formato (shape):", sinal.shape)
print("Primeiros 10 valores:", sinal[:10])


# Visualizando o sinal de vibração para entender o comportamento, comparando com o sinal de falha

dados_falha = loadmat("data/raw/IR021_0.mat")
print(dados_falha.keys())

sinal_falha = dados_falha['X209_DE_time']

fig, axs = plt.subplots(1, 2, figsize=(14, 4))

axs[0].plot(sinal[:2000])
axs[0].set_title("Saudável (Normal_0)")
axs[0].set_ylabel("Aceleração")

axs[1].plot(sinal_falha[:2000])
axs[1].set_title("Falha - Pista Interna 0.021\" (IR021_0)")

plt.tight_layout()
plt.savefig("results/comparando_sinais.png", dpi=150, bbox_inches="tight")
plt.close()

def encontrar_chave_de_time(dados: dict) -> str:
    for chave in dados.keys():
        if "DE_time" in chave:
            return chave
    raise ValueError("Nenhuma chave DE_time encontrada no arquivo.")

# Testando:
chave = encontrar_chave_de_time(dados_falha)
print("Chave encontrada:", chave)

# Montar Dataframe bruto com os sinais de vibração

pasta = "data/raw"
arquivos_com_erro = []

for nome_arquivo in sorted(os.listdir(pasta)):
    if not nome_arquivo.endswith(".mat"):
        continue
    caminho = os.path.join(pasta, nome_arquivo)
    try:
        loadmat(caminho)
    except Exception as e:
        print(f"[CORROMPIDO] {nome_arquivo}: {e}")
        arquivos_com_erro.append(nome_arquivo)

print("\nTotal de arquivos com problema:", len(arquivos_com_erro))
print(arquivos_com_erro)

df = montar_dataframe_bruto("data/raw")
df = preencher_rpm_ausente(df)
print("Shape:", df.shape)
print(df[["arquivo", "classe", "carga", "rpm"]].to_string())

print(df["classe"].value_counts())
print("\nn_amostras - mínimo, máximo, média:")
print(df["n_amostras"].min(), df["n_amostras"].max(), df["n_amostras"].mean())

# Montar dataframe com as janelas

df_janelas = montar_dataframe_janelas(df, tamanho_janela=2048)
print(df_janelas.shape)

#Montar dataframe com features e platar boxplots

df_features = montar_dataframe_features(df_janelas)
print("Shape:", df_features.shape)


sns.boxplot(data=df_features, x="classe", y="rms")
plt.title("RMS por classe")
plt.savefig("results/boxplot_rms.png", dpi=150, bbox_inches="tight")
plt.close()

sns.boxplot(data=df_features, x="classe", y="curtose")
plt.title("Curtose por classe")
plt.savefig("results/boxplot_curtose.png", dpi=150, bbox_inches="tight")
plt.close()

sns.boxplot(data=df_features, x="classe", y="fator_crista")
plt.title("Fator de Crista por classe")
plt.savefig("results/boxplot_fator_crista.png", dpi=150, bbox_inches="tight")
plt.close()

os.makedirs("data/processed", exist_ok=True)
df_features.to_csv("data/processed/features.csv", index=False)

