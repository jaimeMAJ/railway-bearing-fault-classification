"""
03_plot_spectra.py

Geração de figuras de alta resolução (300 DPI) para a monografia do TCC:
- Formas de onda no domínio do tempo (comparativo das 4 classes)
- Espectros de frequência (FFT unilateral com janela de Hanning)
- Espectros de Envelope (Hilbert) com marcação das frequências características de falha (BPFO, BPFI, BSF)

Conforme a Seção 4 (Etapa 2) e Seção 7.4 do plano de TCC (UFSJ).
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    FIGURES_DIR,
    RAW_DATA_DIR,
    RESULTS_DIR,
    TAXA_AMOSTRAGEM,
)
from src.features import (
    calcular_envelope,
    calcular_fft,
    calcular_frequencias_falha,
)
from src.preprocessing import carregar_rpm, carregar_sinal


def main():
    print("=" * 70)
    print("GERAÇÃO DE FIGURAS ESPECTRAIS PARA MONOGRAFIA (300 DPI)")
    print("=" * 70)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Arquivos representativos de cada condição (Carga 0 HP, severidade 0.021" onde aplicável)
    casos = {
        "Normal": ("Normal_0.mat", "Saudável (Baseline)", "#2b5c8f"),
        "IR": ("IR021_0.mat", "Falha na Pista Interna (IR - 0.021\")", "#d95f02"),
        "B": ("B021_0.mat", "Falha em Elemento Rolante (B - 0.021\")", "#7570b3"),
        "OR": ("OR021@6_0.mat", "Falha na Pista Externa (OR - 0.021\")", "#e7298a"),
    }

    # Carregamento dos sinais e RPM
    dados = {}
    for classe, (nome_arq, rotulo, cor) in casos.items():
        caminho = RAW_DATA_DIR / nome_arq
        sinal = carregar_sinal(str(caminho))
        rpm = carregar_rpm(str(caminho))
        if rpm is None:
            rpm = 1797.0  # RPM típica carga 0
        dados[classe] = {
            "sinal": sinal,
            "rpm": rpm,
            "rotulo": rotulo,
            "cor": cor,
            "teoricas": calcular_frequencias_falha(rpm),
        }

    # =========================================================================
    # 1. Comparativo de Formas de Onda no Domínio do Tempo (2048 amostras)
    # =========================================================================
    print("[1/3] Gerando comparativo temporal de aceleração...")
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    tempo_ms = (np.arange(2048) / TAXA_AMOSTRAGEM) * 1000.0  # tempo em milissegundos

    for idx, (classe, info) in enumerate(dados.items()):
        trecho = info["sinal"][:2048]
        axs[idx].plot(tempo_ms, trecho, color=info["cor"], linewidth=0.9)
        axs[idx].set_title(info["rotulo"], fontsize=10.5, fontweight="bold", pad=4)
        axs[idx].set_ylabel("Aceleração [g]", fontsize=9.5)
        axs[idx].grid(True, linestyle="--", alpha=0.5)

    axs[-1].set_xlabel("Tempo [ms]", fontsize=10.5, fontweight="bold")
    plt.suptitle("Sinais de Vibração de Aceleração no Domínio do Tempo (Janela de 2048 pontos)", fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()

    caminho_fig1 = FIGURES_DIR / "comparativo_sinais_tempo.png"
    plt.savefig(caminho_fig1, dpi=300, bbox_inches="tight")
    plt.savefig(RESULTS_DIR / "comparando_sinais.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Salvo em: {caminho_fig1}")

    # =========================================================================
    # 2. Espectro FFT Direto (0 a 3000 Hz)
    # =========================================================================
    print("[2/3] Gerando espectro de frequência FFT (Hanning Window)...")
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    for idx, (classe, info) in enumerate(dados.items()):
        trecho = info["sinal"][:4096]
        freqs, mags = calcular_fft(trecho, TAXA_AMOSTRAGEM, aplicar_hanning=True)
        mask = (freqs <= 3000)

        axs[idx].plot(freqs[mask], mags[mask], color=info["cor"], linewidth=0.85)
        axs[idx].set_title(f"Espectro FFT — {info['rotulo']}", fontsize=10.5, fontweight="bold", pad=4)
        axs[idx].set_ylabel("Magnitude [g]", fontsize=9.5)
        axs[idx].grid(True, linestyle="--", alpha=0.5)

    axs[-1].set_xlabel("Frequência [Hz]", fontsize=10.5, fontweight="bold")
    plt.suptitle("Espectros de Magnitude FFT com Janela de Hanning (0 a 3000 Hz)", fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()

    caminho_fig2 = FIGURES_DIR / "espectros_fft_comparativo.png"
    plt.savefig(caminho_fig2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      Salvo em: {caminho_fig2}")

    # =========================================================================
    # 3. Espectros de Envelope (Hilbert) com Frequências Teóricas
    # =========================================================================
    print("[3/3] Gerando espectros de envelope com frequências características de defeito...")
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    falhas = [
        ("IR", "Falha de Pista Interna (IR021) — Frequência Característica: BPFI", "bpfi", "BPFI", "#d95f02"),
        ("OR", "Falha de Pista Externa (OR021) — Frequência Característica: BPFO", "bpfo", "BPFO", "#e7298a"),
        ("B",  "Falha em Esfera (B021) — Frequência Característica: BSF", "bsf", "BSF", "#7570b3"),
    ]

    for idx, (classe, titulo_sub, chave_freq, nome_freq, cor) in enumerate(falhas):
        info = dados[classe]
        sinal_trecho = info["sinal"][:4096]
        envelope = calcular_envelope(sinal_trecho, remover_dc=True)
        freqs_env, mags_env = calcular_fft(envelope, TAXA_AMOSTRAGEM, aplicar_hanning=True)
        mask_low = (freqs_env <= 500)

        f_alvo = info["teoricas"][chave_freq]

        axs[idx].plot(freqs_env[mask_low], mags_env[mask_low], color=cor, linewidth=1.0, label="Espectro do Envelope")
        axs[idx].axvline(f_alvo, color="black", linestyle="--", linewidth=1.5, label=f"{nome_freq} Teórica ({f_alvo:.1f} Hz)")
        axs[idx].axvline(2 * f_alvo, color="gray", linestyle=":", linewidth=1.2, label=f"2x {nome_freq} ({2*f_alvo:.1f} Hz)")

        axs[idx].set_title(titulo_sub, fontsize=10.5, fontweight="bold", pad=4)
        axs[idx].set_ylabel("Magnitude Envelope", fontsize=9.5)
        axs[idx].legend(loc="upper right", fontsize=8.5)
        axs[idx].grid(True, linestyle="--", alpha=0.5)

    axs[-1].set_xlabel("Frequência de Modulação [Hz]", fontsize=10.5, fontweight="bold")
    plt.suptitle("Análise de Envelope (Transformada de Hilbert) e Frequências de Falha do Rolamento", fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()

    caminho_fig3 = FIGURES_DIR / "espectros_envelope_frequencias_falha.png"
    plt.savefig(caminho_fig3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      Salvo em: {caminho_fig3}")

    print("\nTodas as figuras acadêmicas foram geradas com sucesso!")


if __name__ == "__main__":
    main()
