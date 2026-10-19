"""
01_extract_features.py

Pipeline completo de ingestão de dados brutos (.mat do CWRU), segmentação em janelas,
extração de características estatísticas (tempo e frequência/envelope), validação de qualidade
de dados e persistência na Feature Store (.parquet e .csv).

Conforme as Etapas 1 e 2 das especificações do TCC.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RESULTS_DIR,
    TAMANHO_JANELA_PADRAO,
)
from src.features import montar_dataframe_features, validar_qualidade_features
from src.preprocessing import (
    montar_dataframe_bruto,
    montar_dataframe_janelas,
    preencher_rpm_ausente,
)


def main():
    print("=" * 70)
    print("PIPELINE DE ENGENHARIA DE DADOS E EXTRAÇÃO DE ATRIBUTOS (CWRU)")
    print("=" * 70)

    # 1. Criação de diretórios estruturados
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 2. Ingestão e montagem do DataFrame bruto
    print(f"\n[1/5] Ingerindo arquivos .mat de: {RAW_DATA_DIR}")
    df_bruto = montar_dataframe_bruto(str(RAW_DATA_DIR))
    print(f"      Total de arquivos lidos: {len(df_bruto)}")
    print("      Distribuição de ensaios por classe:")
    for classe, count in df_bruto["classe"].value_counts().items():
        print(f"        - {classe}: {count} arquivos")

    # 3. Tratamento de RPM ausente (imputação com base na carga do motor)
    print("\n[2/5] Validando RPM e aplicando imputação por carga...")
    df_bruto = preencher_rpm_ausente(df_bruto)

    # 4. Segmentação em janelas temporais discretas (com data lineage)
    print(f"\n[3/5] Segmentando séries temporais em janelas de N = {TAMANHO_JANELA_PADRAO} pontos...")
    df_janelas = montar_dataframe_janelas(
        df_bruto,
        tamanho_janela=TAMANHO_JANELA_PADRAO,
        overlap=0.0,
    )
    print(f"      Total de janelas geradas: {len(df_janelas)}")
    print("      Janelas por classe:")
    for classe, count in df_janelas["classe"].value_counts().items():
        print(f"        - {classe}: {count} janelas")

    # 5. Extração de características (Tempo + FFT/Envelope)
    print("\n[4/5] Extraindo atributos nos domínios do tempo e da frequência...")
    df_features = montar_dataframe_features(df_janelas)
    print(f"      Shape da Feature Store: {df_features.shape}")

    # 6. Validação de qualidade de dados (Data Quality Contract)
    print("\n[5/5] Executando testes automatizados de qualidade de dados...")
    validar_qualidade_features(df_features)

    # 7. Persistência dos dados (Parquet + CSV)
    caminho_parquet = PROCESSED_DATA_DIR / "features.parquet"
    caminho_csv = PROCESSED_DATA_DIR / "features.csv"

    print(f"\nSalvando Feature Store em:\n  -> {caminho_parquet}\n  -> {caminho_csv}")
    df_features.to_parquet(caminho_parquet, index=False)
    df_features.to_csv(caminho_csv, index=False)

    # 8. Gráficos de exploração de atributos para a monografia
    print("\nGerando gráficos de análise exploratória em results/figures...")
    sns.set_theme(style="whitegrid")

    features_para_plot = [
        ("rms", "Valor Eficaz (RMS) [g]", "boxplot_rms.png"),
        ("curtose", "Curtose", "boxplot_curtose.png"),
        ("fator_crista", "Fator de Crista", "boxplot_fator_crista.png"),
        ("energia_bpfo", "Energia Espectral BPFO (Envelope)", "boxplot_energia_bpfo.png"),
        ("energia_bpfi", "Energia Espectral BPFI (Envelope)", "boxplot_energia_bpfi.png"),
        ("energia_bsf", "Energia Espectral BSF (Envelope)", "boxplot_energia_bsf.png"),
    ]

    for col, rotulo_y, nome_arq in features_para_plot:
        plt.figure(figsize=(7, 4.5))
        sns.boxplot(data=df_features, x="classe", y=col, hue="classe", legend=False, palette="Blues")
        plt.xlabel("Condição do Rolamento", fontsize=11, fontweight="bold")
        plt.ylabel(rotulo_y, fontsize=11, fontweight="bold")
        plt.title(f"Distribuição de {rotulo_y} por Condição", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / nome_arq, dpi=300, bbox_inches="tight")
        plt.savefig(RESULTS_DIR / nome_arq, dpi=150, bbox_inches="tight")  # compatibilidade
        plt.close()

    print("\nExtração e persistência concluídas com sucesso!")


if __name__ == "__main__":
    main()
