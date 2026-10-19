"""
src/models.py

Módulo de modelagem preditiva, validação cruzada agrupada e avaliação acadêmica
conforme as especificações do TCC (Seção 4: Etapas 3 e 4).
"""

from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import NOMES_CLASSES_PT, RANDOM_STATE

# Definição dos subconjuntos de atributos para análise comparativa
CONJUNTOS_FEATURES = {
    "tempo_base": ["rms", "curtose", "fator_crista"],
    "tempo_completo": [
        "rms", "curtose", "fator_crista", "desvio_padrao",
        "assimetria", "pico_a_pico", "fator_forma", "fator_impulso",
    ],
    "frequencia_envelope": [
        "energia_bpfo", "energia_bpfi", "energia_bsf", "energia_ftf",
        "centroide_espectral", "variancia_espectral",
        "freq_pico_dominante", "amp_pico_dominante",
    ],
    "todas_features": [
        "rms", "curtose", "fator_crista", "desvio_padrao",
        "assimetria", "pico_a_pico", "fator_forma", "fator_impulso",
        "energia_bpfo", "energia_bpfi", "energia_bsf", "energia_ftf",
        "centroide_espectral", "variancia_espectral",
        "freq_pico_dominante", "amp_pico_dominante",
    ],
}


def dividir_treino_teste_por_arquivo(
    df: pd.DataFrame,
    coluna_alvo: str = "classe",
    test_size: float = 0.3,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide os dados em conjuntos de treino e teste no nível de ARQUIVO (ensaio),
    garantindo a completa prevenção de vazamento de dados (data leakage) entre janelas.
    """
    arquivos_unicos = df[["arquivo", coluna_alvo]].drop_duplicates().reset_index(drop=True)

    arquivos_treino, arquivos_teste = train_test_split(
        arquivos_unicos["arquivo"],
        test_size=test_size,
        stratify=arquivos_unicos[coluna_alvo],
        random_state=random_state,
    )

    df_treino = df[df["arquivo"].isin(arquivos_treino)].copy()
    df_teste = df[df["arquivo"].isin(arquivos_teste)].copy()

    return df_treino, df_teste


def criar_modelo_rf(random_state: int = RANDOM_STATE, **kwargs) -> RandomForestClassifier:
    """Instancia o classificador Random Forest configurado para o projeto."""
    parametros = {
        "n_estimators": 100,
        "max_depth": kwargs.get("max_depth", None),
        "min_samples_split": kwargs.get("min_samples_split", 2),
        "criterion": kwargs.get("criterion", "gini"),
        "random_state": random_state,
        "n_jobs": -1,
    }
    parametros.update(kwargs)
    return RandomForestClassifier(**parametros)


def criar_modelo_svm(random_state: int = RANDOM_STATE, **kwargs) -> Pipeline:
    """
    Instancia pipeline SVM com padronização via StandardScaler ajustada
    estritamente nos dados de treinamento.
    """
    kernel = kwargs.get("kernel", "rbf")
    c_val = kwargs.get("C", 1.0)
    gamma = kwargs.get("gamma", "scale")

    return make_pipeline(
        StandardScaler(),
        SVC(
            C=c_val,
            kernel=kernel,
            gamma=gamma,
            random_state=random_state,
        ),
    )


def executar_validacao_cruzada_agrupada(
    df: pd.DataFrame,
    modelo_tipo: str = "RF",
    colunas_features: List[str] = None,
    coluna_alvo: str = "classe",
    n_splits: int = 4,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Executa Validação Cruzada Estratificada Agrupada por Arquivo (StratifiedGroupKFold).
    Retorna tabela de resultados por fold com Acurácia, F1 Macro e Recall por classe.
    """
    if colunas_features is None:
        colunas_features = CONJUNTOS_FEATURES["todas_features"]

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X = df[colunas_features]
    y = df[coluna_alvo]
    grupos = df["arquivo"]

    resultados_folds = []

    for fold, (idx_treino, idx_teste) in enumerate(sgkf.split(X, y, grupos)):
        X_tr, y_tr = X.iloc[idx_treino], y.iloc[idx_treino]
        X_te, y_te = X.iloc[idx_teste], y.iloc[idx_teste]

        if modelo_tipo.upper() == "RF":
            modelo = criar_modelo_rf(random_state=random_state)
        elif modelo_tipo.upper() == "SVM":
            modelo = criar_modelo_svm(random_state=random_state)
        else:
            raise ValueError(f"Tipo de modelo desconhecido: {modelo_tipo}")

        modelo.fit(X_tr, y_tr)
        y_pred = modelo.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        f1_mac = f1_score(y_te, y_pred, average="macro")
        f1_wei = f1_score(y_te, y_pred, average="weighted")

        registro = {
            "modelo": modelo_tipo,
            "fold": fold,
            "acuracia": acc,
            "f1_macro": f1_mac,
            "f1_weighted": f1_wei,
        }

        # Recall específico de cada classe no fold
        for classe in sorted(y.unique()):
            mask_classe = (y_te == classe)
            if mask_classe.sum() > 0:
                rec = recall_score(y_te == classe, y_pred == classe)
                registro[f"recall_{classe}"] = rec

        resultados_folds.append(registro)

    return pd.DataFrame(resultados_folds)


def avaliar_modelo(
    modelo,
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
    X_teste: pd.DataFrame,
    y_teste: pd.Series,
) -> Dict:
    """Treina o modelo e calcula métricas completas no conjunto de teste."""
    modelo.fit(X_treino, y_treino)
    y_pred = modelo.predict(X_teste)

    labels = sorted(y_teste.unique())
    matriz = confusion_matrix(y_teste, y_pred, labels=labels)
    matriz_norm = confusion_matrix(y_teste, y_pred, labels=labels, normalize="true")

    relatorio_dict = classification_report(y_teste, y_pred, output_dict=True, zero_division=0)
    relatorio_txt = classification_report(y_teste, y_pred, zero_division=0)

    return {
        "modelo": modelo,
        "y_pred": y_pred,
        "labels": labels,
        "acuracia": accuracy_score(y_teste, y_pred),
        "f1_macro": f1_score(y_teste, y_pred, average="macro"),
        "f1_weighted": f1_score(y_teste, y_pred, average="weighted"),
        "matriz_confusao": matriz,
        "matriz_confusao_norm": matriz_norm,
        "relatorio_dict": relatorio_dict,
        "relatorio_texto": relatorio_txt,
    }


def votar_por_arquivo(
    df_teste: pd.DataFrame,
    y_pred: np.ndarray,
    coluna_alvo: str = "classe",
) -> pd.DataFrame:
    """
    Aplica mecanismo de votação majoritária por ensaio (File-level Majority Voting).
    Determina a classe predita global para cada arquivo físico de rolamento.
    """
    res = df_teste[["arquivo", coluna_alvo]].copy()
    res["predito"] = y_pred

    agrupado = []
    for arquivo, grupo in res.groupby("arquivo"):
        classe_real = grupo[coluna_alvo].iloc[0]
        voto_majoritario = grupo["predito"].mode()[0]
        concordancia = (grupo["predito"] == voto_majoritario).mean()
        acertou = (voto_majoritario == classe_real)

        agrupado.append({
            "arquivo": arquivo,
            "classe_real": classe_real,
            "classe_predita": voto_majoritario,
            "concordancia_votos": concordancia,
            "acertou": acertou,
            "total_janelas": len(grupo),
        })

    return pd.DataFrame(agrupado)


def plotar_matriz_confusao(
    matriz: np.ndarray,
    labels: List[str],
    titulo: str,
    caminho_saida: str,
    normalizada: bool = False,
    dpi: int = 300,
) -> None:
    """
    Plota e exporta matriz de confusão em alta resolução (300 DPI) com visual acadêmico,
    seguindo o padrão para inserção direta na monografia de TCC.
    """
    plt.figure(figsize=(7, 5.5))

    fmt = ".2%" if normalizada else "d"
    vmax = 1.0 if normalizada else None

    # Mapeamento para nomes descritivos
    rotulos_exibicao = [NOMES_CLASSES_PT.get(l, l) for l in labels]

    sns.heatmap(
        matriz,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=rotulos_exibicao,
        yticklabels=rotulos_exibicao,
        cbar=True,
        vmax=vmax,
        annot_kws={"size": 11, "weight": "bold"},
    )

    plt.xlabel("Classe Predita pelo Modelo", fontsize=11, fontweight="bold")
    plt.ylabel("Classe Real (Condição Física)", fontsize=11, fontweight="bold")
    plt.title(titulo, fontsize=12, fontweight="bold", pad=12)
    plt.xticks(rotation=25, ha="right", fontsize=9.5)
    plt.yticks(rotation=0, fontsize=9.5)
    plt.tight_layout()

    plt.savefig(caminho_saida, dpi=dpi, bbox_inches="tight")
    plt.close()


def plotar_importancia_features(
    modelo_rf: RandomForestClassifier,
    nomes_features: List[str],
    caminho_saida: str,
    top_n: int = 15,
    dpi: int = 300,
) -> pd.DataFrame:
    """
    Gera gráfico de barras das features mais importantes (MDI) para interpretabilidade
    física e mecânica do diagnóstico no TCC.
    """
    importancias = modelo_rf.feature_importances_
    df_imp = pd.DataFrame({
        "feature": nomes_features,
        "importancia": importancias,
    }).sort_values(by="importancia", ascending=False).reset_index(drop=True)

    df_top = df_imp.head(top_n)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df_top,
        x="importancia",
        y="feature",
        hue="feature",
        legend=False,
        palette="viridis",
    )

    plt.xlabel("Importância Média (Gini Importance)", fontsize=11, fontweight="bold")
    plt.ylabel("Atributo Extraído", fontsize=11, fontweight="bold")
    plt.title("Interpretabilidade Mecânica: Relevância dos Atributos (Random Forest)", fontsize=12, fontweight="bold", pad=10)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig(caminho_saida, dpi=dpi, bbox_inches="tight")
    plt.close()

    return df_imp
