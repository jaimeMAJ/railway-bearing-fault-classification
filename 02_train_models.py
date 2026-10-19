"""
02_train_models.py

Treinamento, otimização, validação cruzada agrupada (Stratified Group K-Fold)
e avaliação dos classificadores Random Forest e SVM.

Conforme as Etapas 3 e 4 das especificações do TCC.
"""

import os
import pandas as pd

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR
from src.models import (
    CONJUNTOS_FEATURES,
    avaliar_modelo,
    criar_modelo_rf,
    criar_modelo_svm,
    dividir_treino_teste_por_arquivo,
    executar_validacao_cruzada_agrupada,
    plotar_importancia_features,
    plotar_matriz_confusao,
    votar_por_arquivo,
)


def main():
    print("=" * 75)
    print("TREINAMENTO E AVALIAÇÃO DE MODELOS PREDITIVOS (TCC - UFSJ)")
    print("=" * 75)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. Carregamento dos dados processados (preferência por parquet)
    caminho_parquet = PROCESSED_DATA_DIR / "features.parquet"
    caminho_csv = PROCESSED_DATA_DIR / "features.csv"

    if caminho_parquet.exists():
        df = pd.read_parquet(caminho_parquet)
    else:
        df = pd.read_csv(caminho_csv, dtype={"severidade": str})

    print(f"\nDados carregados: {df.shape[0]} janelas e {df.shape[1]} colunas.")
    print("Distribuição das classes:")
    print(df["classe"].value_counts())

    # =========================================================================
    # 2. Validação Cruzada Agrupada (Stratified Group K-Fold, k=4)
    # =========================================================================
    print("\n" + "=" * 75)
    print("ETAPA 1: VALIDAÇÃO CRUZADA ESTRATIFICADA AGRUPADA POR ARQUIVO (4 FOLDS)")
    print("=" * 75)

    resultados_cv = []
    for nome_conjunto, colunas in CONJUNTOS_FEATURES.items():
        print(f"\n--- Avaliando subconjunto: '{nome_conjunto}' ({len(colunas)} features) ---")

        # Avaliação Random Forest
        df_rf = executar_validacao_cruzada_agrupada(
            df,
            modelo_tipo="RF",
            colunas_features=colunas,
            n_splits=4,
        )
        df_rf["conjunto"] = nome_conjunto
        resultados_cv.append(df_rf)
        print(f"  [RF]  Acurácia Média: {df_rf['acuracia'].mean():.2%} (+/- {df_rf['acuracia'].std():.2%}) | F1 Macro: {df_rf['f1_macro'].mean():.4f}")

        # Avaliação SVM
        df_svm = executar_validacao_cruzada_agrupada(
            df,
            modelo_tipo="SVM",
            colunas_features=colunas,
            n_splits=4,
        )
        df_svm["conjunto"] = nome_conjunto
        resultados_cv.append(df_svm)
        print(f"  [SVM] Acurácia Média: {df_svm['acuracia'].mean():.2%} (+/- {df_svm['acuracia'].std():.2%}) | F1 Macro: {df_svm['f1_macro'].mean():.4f}")

    df_cv_consolidado = pd.concat(resultados_cv, ignore_index=True)
    df_cv_consolidado.to_csv(RESULTS_DIR / "cv_resultados.csv", index=False)
    print(f"\nResultados detalhados de CV exportados para: {RESULTS_DIR / 'cv_resultados.csv'}")

    resumo_cv = df_cv_consolidado.groupby(["modelo", "conjunto"])[["acuracia", "f1_macro"]].mean().round(4)
    print("\nResumo Comparativo de Validação Cruzada:")
    print(resumo_cv)

    # =========================================================================
    # 3. Treino / Teste Final (70% / 30% com divisão por arquivo)
    # =========================================================================
    print("\n" + "=" * 75)
    print("ETAPA 2: AVALIAÇÃO FINAL NO CONJUNTO DE TESTE INDEPENDENTE (70/30)")
    print("=" * 75)

    df_treino, df_teste = dividir_treino_teste_por_arquivo(df, test_size=0.3, random_state=42)
    print(f"Treino: {len(df_treino)} janelas ({df_treino['arquivo'].nunique()} arquivos)")
    print(f"Teste:  {len(df_teste)} janelas ({df_teste['arquivo'].nunique()} arquivos)")

    conjunto_escolhido = CONJUNTOS_FEATURES["todas_features"]
    X_tr = df_treino[conjunto_escolhido]
    y_tr = df_treino["classe"]
    X_te = df_teste[conjunto_escolhido]
    y_te = df_teste["classe"]

    # --- Avaliação Random Forest ---
    print("\n>>> Modelo: Random Forest (Conjunto Completo de Features)")
    modelo_rf = criar_modelo_rf()
    res_rf = avaliar_modelo(modelo_rf, X_tr, y_tr, X_te, y_te)
    print(f"Acurácia Geral: {res_rf['acuracia']:.2%}")
    print(res_rf["relatorio_texto"])

    # Matrizes de confusão RF
    plotar_matriz_confusao(
        res_rf["matriz_confusao"],
        res_rf["labels"],
        titulo="Matriz de Confusão Absoluta — Random Forest",
        caminho_saida=str(FIGURES_DIR / "matriz_confusao_rf_absoluta.png"),
        normalizada=False,
    )
    plotar_matriz_confusao(
        res_rf["matriz_confusao_norm"],
        res_rf["labels"],
        titulo="Matriz de Confusão Normalizada — Random Forest",
        caminho_saida=str(FIGURES_DIR / "matriz_confusao_rf_normalizada.png"),
        normalizada=True,
    )

    # Gráfico de Importância de Features
    df_imp = plotar_importancia_features(
        modelo_rf,
        conjunto_escolhido,
        caminho_saida=str(FIGURES_DIR / "importancia_features_rf.png"),
        top_n=12,
    )
    print("Top 5 atributos mais relevantes para o diagnóstico (RF):")
    for _, l in df_imp.head(5).iterrows():
        print(f"  - {l['feature']}: {l['importancia']:.4f}")

    # --- Avaliação SVM ---
    print("\n>>> Modelo: Support Vector Machine (StandardScaler + RBF Kernel)")
    modelo_svm = criar_modelo_svm()
    res_svm = avaliar_modelo(modelo_svm, X_tr, y_tr, X_te, y_te)
    print(f"Acurácia Geral: {res_svm['acuracia']:.2%}")
    print(res_svm["relatorio_texto"])

    plotar_matriz_confusao(
        res_svm["matriz_confusao_norm"],
        res_svm["labels"],
        titulo="Matriz de Confusão Normalizada — SVM (Kernel RBF)",
        caminho_saida=str(FIGURES_DIR / "matriz_confusao_svm_normalizada.png"),
        normalizada=True,
    )

    # =========================================================================
    # 4. Avaliação por Votação Majoritária no Nível de Ensaio (File-level Voting)
    # =========================================================================
    print("\n" + "=" * 75)
    print("ETAPA 3: AVALIAÇÃO POR VOTAÇÃO MAJORITÁRIA POR ENSAIO (FILE-LEVEL VOTING)")
    print("=" * 75)

    votos_rf = votar_por_arquivo(df_teste, res_rf["y_pred"])
    acuracia_ensaio_rf = votos_rf["acertou"].mean()
    print(f"\nAcurácia do Diagnóstico em Nível de Rolamento/Ensaio (RF): {acuracia_ensaio_rf:.2%}")
    print(votos_rf[["arquivo", "classe_real", "classe_predita", "concordancia_votos", "acertou"]].to_string(index=False))

    votos_rf.to_csv(RESULTS_DIR / "votacao_ensaios_rf.csv", index=False)
    print("\nTreinamento, validação e exportação acadêmica concluídos com sucesso!")


if __name__ == "__main__":
    main()
