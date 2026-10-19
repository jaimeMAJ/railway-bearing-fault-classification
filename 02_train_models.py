import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Preparar os dados para treinamento e teste

df_features = pd.read_csv("data/processed/features.csv", dtype={"severidade": str})
arquivos_unicos = df_features[["arquivo", "classe"]].drop_duplicates()
print(arquivos_unicos.shape)

arquivos_treino, arquivos_teste = train_test_split(
    arquivos_unicos["arquivo"],
    test_size=0.3,
    stratify=arquivos_unicos["classe"],
    random_state=42,
)

print("Arquivos treino:", len(arquivos_treino))
print("Arquivos teste:", len(arquivos_teste))

df_treino = df_features[df_features["arquivo"].isin(arquivos_treino)]
df_teste = df_features[df_features["arquivo"].isin(arquivos_teste)]

print("Janelas treino:", df_treino.shape[0])
print("Janelas teste:", df_teste.shape[0])

base = ["rms", "curtose", "fator_crista"]
conjuntos = {
    "base_3": base,
    "base_bpfo": base + ["energia_bpfo"],
    "base_bpfi": base + ["energia_bpfi"],
    "base_bsf": base + ["energia_bsf"],
    "base_todas": base + ["energia_bpfo", "energia_bpfi", "energia_bsf"],
}

for nome, colunas_features in conjuntos.items():
    print(f"\n{'=' * 60}\nConjunto: {nome}\n{'=' * 60}")

    X_treino, y_treino = df_treino[colunas_features], df_treino["classe"]
    X_teste, y_teste = df_teste[colunas_features], df_teste["classe"]

    modelo_rf = RandomForestClassifier(random_state=42)
    modelo_rf.fit(X_treino, y_treino)
    y_previsto = modelo_rf.predict(X_teste)

    resultado = df_teste[["arquivo", "classe"]].copy()
    resultado["acertou"] = (y_previsto == y_teste.values)
    resultado["previsto"] = y_previsto
    print(resultado.groupby(["arquivo", "previsto"]).size().unstack(fill_value=0))
    print(resultado.groupby(["classe", "arquivo"])["acertou"].mean().round(2))

    print(f"Acurácia: {accuracy_score(y_teste, y_previsto):.2%}")
    print(classification_report(y_teste, y_previsto))

    print("Importância das features:")
    for feature, importancia in zip(colunas_features, modelo_rf.feature_importances_):
        print(f"  {feature}: {importancia:.3f}")

    matriz = confusion_matrix(y_teste, y_previsto, labels=modelo_rf.classes_)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues",
                xticklabels=modelo_rf.classes_, yticklabels=modelo_rf.classes_)
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusão - RF ({nome})")
    plt.savefig(f"results/matriz_confusao_rf_{nome}.png", dpi=150, bbox_inches="tight")
    plt.close()

colunas = ["rms", "curtose", "fator_crista", "energia_bpfo", "energia_bpfi", "energia_bsf"]
print(df_features.groupby(["classe", "arquivo"])[colunas].mean().round(1))