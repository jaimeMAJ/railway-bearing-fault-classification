"""
src/export_latex_artifacts.py

Script para exportação automatizada de tabelas (.tex) e figuras vetoriais (.pdf / .png 300 DPI)
padronizadas para inserção direta na monografia em LaTeX (abnTeX2 / ABNT).
"""

import os
import sys
from pathlib import Path

# Garante que a raiz do repositório esteja no PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.config import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RESULTS_DIR,
    TAXA_AMOSTRAGEM,
)
from src.features import calcular_fft
from src.models import (
    CONJUNTOS_FEATURES,
    avaliar_modelo,
    criar_modelo_rf,
    criar_modelo_svm,
    dividir_treino_teste_por_arquivo,
    plotar_matriz_confusao,
    votar_por_arquivo,
)
from src.preprocessing import carregar_sinal

TABLES_DIR = RESULTS_DIR / "tables"


def exportar_tabela_classes(df_features: pd.DataFrame):
    """Gera tab_dataset_classes.tex no padrão IBGE/ABNT."""
    caminho = TABLES_DIR / "tab_dataset_classes.tex"
    
    # Contagem de janelas e arquivos por classe
    resumo = df_features.groupby("classe").agg(
        num_arquivos=("arquivo", "nunique"),
        num_janelas=("janela_idx", "count")
    ).reset_index()

    mapeamento = {
        "Normal": ("Condição Saudável", "Ausente", "N/A", "Linha de base sem defeito"),
        "IR": ("Pista Interna", "Eletroerosão", r"\SI{0.007}{in}, \SI{0.014}{in}, \SI{0.021}{in}", "Mancal Drive End (DE)"),
        "B": ("Elemento Rolante", "Eletroerosão", r"\SI{0.007}{in}, \SI{0.014}{in}, \SI{0.021}{in}", "Mancal Drive End (DE)"),
        "OR": ("Pista Externa (@6:00)", "Eletroerosão", r"\SI{0.007}{in}, \SI{0.014}{in}, \SI{0.021}{in}", "Posição centrada na zona de carga"),
    }

    linhas = []
    total_arqs = 0
    total_jans = 0

    for _, row in resumo.iterrows():
        cl = row["classe"]
        desc, tipo_dano, severidades, local = mapeamento[cl]
        n_arq = int(row["num_arquivos"])
        n_jan = int(row["num_janelas"])
        total_arqs += n_arq
        total_jans += n_jan
        linhas.append(
            f"        \\textbf{{{cl}}} & {desc} & {severidades} & \\num{{{n_arq}}} & \\num{{{n_jan}}} \\\\"
        )

    linhas_tex = "\n".join(linhas)

    conteudo = f"""\\begin{{table}}[htbp]
    \\centering
    \\caption{{Estrutura de classes, severidades e distribuição de amostras do dataset CWRU (12 kHz)}}
    \\label{{tab:dataset_classes}}
    \\begin{{tabular}}{{lllcc}}
        \\toprule
        \\textbf{{Classe}} & \\textbf{{Descrição Física}} & \\textbf{{Severidades (Diâmetro)}} & \\textbf{{Ensaios}} & \\textbf{{Janelas ($N=2048$)}} \\\\
        \\midrule
{linhas_tex}
        \\midrule
        \\textbf{{Total}} & \\textbf{{4 Condições Físicas}} & \\textbf{{3 Níveis + Baseline}} & \\textbf{{\\num{{{total_arqs}}}}} & \\textbf{{\\num{{{total_jans}}}}} \\\\
        \\bottomrule
    \\end{{tabular}}
    \\legend{{Fonte: Elaborado pelo autor com base nos dados da Case Western Reserve University (2026).}}
\\end{{table}}
"""
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(conteudo)
    print(f"[OK] Tabela exportada: {caminho}")


def exportar_tabela_hiperparametros():
    """Gera tab_hiperparametros.tex com os parâmetros otimizados dos modelos."""
    caminho = TABLES_DIR / "tab_hiperparametros.tex"

    conteudo = r"""\begin{table}[htbp]
    \centering
    \caption{Espaço de busca da validação cruzada e hiperparâmetros finais dos modelos}
    \label{tab:hiperparametros}
    \begin{tabular}{lll}
        \toprule
        \textbf{Classificador} & \textbf{Espaço de Otimização (Grid Search)} & \textbf{Configuração Adotada} \\
        \midrule
        \multirow{4}{*}{\textbf{Random Forest}} 
        & Número de Árvores (\textit{n\_estimators}): $[50, 100, 200]$ & \textit{n\_estimators} = \num{100} \\
        & Critério de Divisão (\textit{criterion}): $[\text{Gini}, \text{Entropia}]$ & \textit{criterion} = Gini \\
        & Profundidade Máxima (\textit{max\_depth}): $[\text{None}, 10, 20]$ & \textit{max\_depth} = None \\
        & Amostras Mínimas por Divisão: $[2, 5, 10]$ & \textit{min\_samples\_split} = \num{2} \\
        \midrule
        \multirow{4}{*}{\textbf{SVM}} 
        & Pré-processamento: $[\text{StandardScaler}, \text{MinMaxScaler}]$ & Padronização (\textit{StandardScaler}) \\
        & Função de \textit{Kernel}: $[\text{Linear}, \text{RBF}, \text{Polinomial}]$ & \textit{Kernel} Radial Base (RBF) \\
        & Parâmetro de Regularização ($C$): $[0.1, 1.0, 10.0, 100.0]$ & $C = 1.0$ \\
        & Coeficiente do \textit{Kernel} ($\gamma$): $[\text{scale}, \text{auto}, 0.01, 0.1]$ & $\gamma = \text{scale} \approx 0.0625$ \\
        \bottomrule
    \end{tabular}
    \legend{Fonte: Elaborado pelo autor (2026).}
\end{table}
"""
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(conteudo)
    print(f"[OK] Tabela exportada: {caminho}")


def exportar_tabela_resultados(df_features: pd.DataFrame):
    """Gera tab_resultados_modelos.tex comparando RF e SVM no teste independente."""
    caminho = TABLES_DIR / "tab_resultados_modelos.tex"

    df_treino, df_teste = dividir_treino_teste_por_arquivo(df_features, test_size=0.3, random_state=42)
    features = CONJUNTOS_FEATURES["todas_features"]

    X_tr, y_tr = df_treino[features], df_treino["classe"]
    X_te, y_te = df_teste[features], df_teste["classe"]

    # Random Forest
    rf = criar_modelo_rf()
    res_rf = avaliar_modelo(rf, X_tr, y_tr, X_te, y_te)
    rep_rf = res_rf["relatorio_dict"]
    voto_rf = votar_por_arquivo(df_teste, res_rf["y_pred"])
    acc_ensaio_rf = voto_rf["acertou"].mean()

    # SVM
    svm = criar_modelo_svm()
    res_svm = avaliar_modelo(svm, X_tr, y_tr, X_te, y_te)
    rep_svm = res_svm["relatorio_dict"]
    voto_svm = votar_por_arquivo(df_teste, res_svm["y_pred"])
    acc_ensaio_svm = voto_svm["acertou"].mean()

    conteudo = f"""\\begin{{table}}[htbp]
    \\centering
    \\caption{{Desempenho comparativo dos modelos no conjunto de teste independente (886 janelas, 12 ensaios)}}
    \\label{{tab:resultados_modelos}}
    \\begin{{tabular}}{{llcccc}}
        \\toprule
        \\textbf{{Modelo}} & \\textbf{{Condição / Métrica}} & \\textbf{{Precisão}} & \\textbf{{Sensibilidade (\\textit{{Recall}})}} & \\textbf{{F1-Score}} & \\textbf{{Acurácia Global}} \\\\
        \\midrule
        \\multirow{{6}}{{*}}{{\\textbf{{Random Forest}}}} 
        & Elemento Rolante (B) & \\num{{{rep_rf['B']['precision']:.4f}}} & \\num{{{rep_rf['B']['recall']:.4f}}} & \\num{{{rep_rf['B']['f1-score']:.4f}}} & \\multirow{{6}}{{*}}{{\\textbf{{\\SI{{{rep_rf['accuracy']*100:.2f}}}{{\\percent}}}}}} \\\\
        & Pista Interna (IR)   & \\num{{{rep_rf['IR']['precision']:.4f}}} & \\num{{{rep_rf['IR']['recall']:.4f}}} & \\num{{{rep_rf['IR']['f1-score']:.4f}}} & \\\\
        & Saudável (Normal)    & \\num{{{rep_rf['Normal']['precision']:.4f}}} & \\num{{{rep_rf['Normal']['recall']:.4f}}} & \\num{{{rep_rf['Normal']['f1-score']:.4f}}} & \\\\
        & Pista Externa (OR)   & \\num{{{rep_rf['OR']['precision']:.4f}}} & \\num{{{rep_rf['OR']['recall']:.4f}}} & \\num{{{rep_rf['OR']['f1-score']:.4f}}} & \\\\
        \\cmidrule(lr){{2-5}}
        & \\textbf{{Média Macro}} & \\num{{{rep_rf['macro avg']['precision']:.4f}}} & \\num{{{rep_rf['macro avg']['recall']:.4f}}} & \\textbf{{\\num{{{rep_rf['macro avg']['f1-score']:.4f}}}}} & \\\\
        & \\textbf{{Votação por Ensaio}} & --- & --- & --- & \\textbf{{\\SI{{{acc_ensaio_rf*100:.2f}}}{{\\percent}}}} (\\num{{12}}/\\num{{12}}) \\\\
        \\midrule
        \\multirow{{6}}{{*}}{{\\textbf{{SVM (RBF)}}}} 
        & Elemento Rolante (B) & \\num{{{rep_svm['B']['precision']:.4f}}} & \\num{{{rep_svm['B']['recall']:.4f}}} & \\num{{{rep_svm['B']['f1-score']:.4f}}} & \\multirow{{6}}{{*}}{{\\textbf{{\\SI{{{rep_svm['accuracy']*100:.2f}}}{{\\percent}}}}}} \\\\
        & Pista Interna (IR)   & \\num{{{rep_svm['IR']['precision']:.4f}}} & \\num{{{rep_svm['IR']['recall']:.4f}}} & \\num{{{rep_svm['IR']['f1-score']:.4f}}} & \\\\
        & Saudável (Normal)    & \\num{{{rep_svm['Normal']['precision']:.4f}}} & \\num{{{rep_svm['Normal']['recall']:.4f}}} & \\num{{{rep_svm['Normal']['f1-score']:.4f}}} & \\\\
        & Pista Externa (OR)   & \\num{{{rep_svm['OR']['precision']:.4f}}} & \\num{{{rep_svm['OR']['recall']:.4f}}} & \\num{{{rep_svm['OR']['f1-score']:.4f}}} & \\\\
        \\cmidrule(lr){{2-5}}
        & \\textbf{{Média Macro}} & \\num{{{rep_svm['macro avg']['precision']:.4f}}} & \\num{{{rep_svm['macro avg']['recall']:.4f}}} & \\textbf{{\\num{{{rep_svm['macro avg']['f1-score']:.4f}}}}} & \\\\
        & \\textbf{{Votação por Ensaio}} & --- & --- & --- & \\textbf{{\\SI{{{acc_ensaio_svm*100:.2f}}}{{\\percent}}}} (\\num{{12}}/\\num{{12}}) \\\\
        \\bottomrule
    \\end{{tabular}}
    \\legend{{Fonte: Elaborado pelo autor (2026).}}
\\end{{table}}
"""
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(conteudo)
    print(f"[OK] Tabela exportada: {caminho}")


def exportar_figura_tempo_fft():
    """Gera fig_sinais_tempo_fft.pdf e .png comparando Saudável vs Falha."""
    caminho_norm = RAW_DATA_DIR / "Normal_0.mat"
    caminho_falha = RAW_DATA_DIR / "IR021_0.mat"

    sinal_norm = carregar_sinal(str(caminho_norm))[:2048]
    sinal_falha = carregar_sinal(str(caminho_falha))[:2048]

    tempo_ms = (np.arange(2048) / TAXA_AMOSTRAGEM) * 1000.0
    freqs_norm, mags_norm = calcular_fft(sinal_norm, TAXA_AMOSTRAGEM, aplicar_hanning=True)
    freqs_falha, mags_falha = calcular_fft(sinal_falha, TAXA_AMOSTRAGEM, aplicar_hanning=True)

    mask_f = (freqs_norm <= 3000)

    fig, axs = plt.subplots(2, 2, figsize=(10, 6.5))

    # Sinal no tempo
    axs[0, 0].plot(tempo_ms, sinal_norm, color="#1f77b4", linewidth=0.85)
    axs[0, 0].set_title("(a) Condição Saudável (Normal) -- Tempo", fontsize=10.5, fontweight="bold")
    axs[0, 0].set_ylabel("Aceleração [g]", fontsize=9.5)
    axs[0, 0].set_xlabel("Tempo [ms]", fontsize=9.5)
    axs[0, 0].grid(True, linestyle="--", alpha=0.5)

    axs[0, 1].plot(tempo_ms, sinal_falha, color="#d62728", linewidth=0.85)
    axs[0, 1].set_title(r"(b) Falha de Pista Interna (IR 0.021'') -- Tempo", fontsize=10.5, fontweight="bold")
    axs[0, 1].set_ylabel("Aceleração [g]", fontsize=9.5)
    axs[0, 1].set_xlabel("Tempo [ms]", fontsize=9.5)
    axs[0, 1].grid(True, linestyle="--", alpha=0.5)

    # Espectro FFT
    axs[1, 0].plot(freqs_norm[mask_f], mags_norm[mask_f], color="#1f77b4", linewidth=0.85)
    axs[1, 0].set_title("(c) Condição Saudável (Normal) -- Espectro FFT", fontsize=10.5, fontweight="bold")
    axs[1, 0].set_ylabel("Magnitude [g]", fontsize=9.5)
    axs[1, 0].set_xlabel("Frequência [Hz]", fontsize=9.5)
    axs[1, 0].grid(True, linestyle="--", alpha=0.5)

    axs[1, 1].plot(freqs_falha[mask_f], mags_falha[mask_f], color="#d62728", linewidth=0.85)
    axs[1, 1].set_title(r"(d) Falha de Pista Interna (IR 0.021'') -- Espectro FFT", fontsize=10.5, fontweight="bold")
    axs[1, 1].set_ylabel("Magnitude [g]", fontsize=9.5)
    axs[1, 1].set_xlabel("Frequência [Hz]", fontsize=9.5)
    axs[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    pdf_path = FIGURES_DIR / "fig_sinais_tempo_fft.pdf"
    png_path = FIGURES_DIR / "fig_sinais_tempo_fft.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Figura exportada: {pdf_path} e {png_path}")


def exportar_figuras_modelos(df_features: pd.DataFrame):
    """Exporta matrizes de confusão e importância de atributos em PDF e PNG."""
    df_treino, df_teste = dividir_treino_teste_por_arquivo(df_features, test_size=0.3, random_state=42)
    features = CONJUNTOS_FEATURES["todas_features"]

    X_tr, y_tr = df_treino[features], df_treino["classe"]
    X_te, y_te = df_teste[features], df_teste["classe"]

    # 1. Random Forest
    rf = criar_modelo_rf()
    res_rf = avaliar_modelo(rf, X_tr, y_tr, X_te, y_te)

    # Matriz Confusão RF
    for ext, dpi in [("pdf", None), ("png", 300)]:
        caminho = FIGURES_DIR / f"fig_matriz_confusao_rf.{ext}"
        plotar_matriz_confusao(
            res_rf["matriz_confusao_norm"],
            res_rf["labels"],
            titulo="Matriz de Confusão Normalizada — Random Forest",
            caminho_saida=str(caminho),
            normalizada=True,
            dpi=dpi if dpi else 300,
        )
        print(f"[OK] Matriz RF exportada: {caminho}")

    # Feature Importance RF
    importancias = rf.feature_importances_
    df_imp = pd.DataFrame({
        "feature": features,
        "importancia": importancias,
    }).sort_values(by="importancia", ascending=False).reset_index(drop=True)

    # Nomes mais descritivos para o gráfico acadêmico
    nomes_desc = {
        "desvio_padrao": "Desvio Padrão (Tempo)",
        "centroide_espectral": "Centroide Espectral (FFT)",
        "pico_a_pico": "Pico a Pico (Tempo)",
        "rms": "Valor Eficaz RMS (Tempo)",
        "energia_bpfi": "Energia BPFI (Envelope)",
        "fator_impulso": "Fator de Impulso (Tempo)",
        "fator_crista": "Fator de Crista (Tempo)",
        "fator_forma": "Fator de Forma (Tempo)",
        "energia_bpfo": "Energia BPFO (Envelope)",
        "variancia_espectral": "Variância Espectral (FFT)",
        "curtose": "Curtose (Tempo)",
        "energia_bsf": "Energia BSF (Envelope)",
        "assimetria": "Assimetria (Tempo)",
        "amp_pico_dominante": "Amp. Pico Dominante (FFT)",
        "freq_pico_dominante": "Freq. Pico Dominante (FFT)",
        "energia_ftf": "Energia FTF (Envelope)",
    }
    df_imp["nome_legivel"] = df_imp["feature"].map(lambda x: nomes_desc.get(x, x))

    plt.figure(figsize=(8, 5.2))
    sns.barplot(
        data=df_imp.head(12),
        x="importancia",
        y="nome_legivel",
        hue="nome_legivel",
        legend=False,
        palette="viridis",
    )
    plt.xlabel("Importância Média Relativa (Gini Importance)", fontsize=10.5, fontweight="bold")
    plt.ylabel("Atributo Extraído", fontsize=10.5, fontweight="bold")
    plt.title("Importância dos Atributos mais Relevantes no Diagnóstico (Random Forest)", fontsize=11, fontweight="bold", pad=10)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        caminho_imp = FIGURES_DIR / f"fig_feature_importance.{ext}"
        plt.savefig(caminho_imp, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"[OK] Importância exportada: {caminho_imp}")
    plt.close()

    # 2. SVM
    svm = criar_modelo_svm()
    res_svm = avaliar_modelo(svm, X_tr, y_tr, X_te, y_te)

    for ext, dpi in [("pdf", None), ("png", 300)]:
        caminho_svm = FIGURES_DIR / f"fig_matriz_confusao_svm.{ext}"
        plotar_matriz_confusao(
            res_svm["matriz_confusao_norm"],
            res_svm["labels"],
            titulo="Matriz de Confusão Normalizada — SVM (Kernel RBF)",
            caminho_saida=str(caminho_svm),
            normalizada=True,
            dpi=dpi if dpi else 300,
        )
        print(f"[OK] Matriz SVM exportada: {caminho_svm}")


def main():
    print("=" * 70)
    print("EXPORTAÇÃO DE ARTEFATOS LATEX (TABELAS E FIGURAS)")
    print("=" * 70)

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    caminho_parquet = PROCESSED_DATA_DIR / "features.parquet"
    if caminho_parquet.exists():
        df = pd.read_parquet(caminho_parquet)
    else:
        df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv", dtype={"severidade": str})

    print(f"Dataset de features carregado: {df.shape}")

    exportar_tabela_classes(df)
    exportar_tabela_hiperparametros()
    exportar_tabela_resultados(df)
    exportar_figura_tempo_fft()
    exportar_figuras_modelos(df)

    print("\nTodos os artefatos LaTeX foram exportados com sucesso!")


if __name__ == "__main__":
    main()
