"""
Script de download automático do CWRU Bearing Dataset (12 kHz, Drive End).

Baixa os arquivos .mat correspondentes a:
- Baseline (condição saudável)
- Falha na pista interna (IR)
- Falha em elemento rolante / esfera (B)
- Falha na pista externa, posição @6:00 (OR)

para as 3 severidades citadas no plano do TCC (0.007", 0.014", 0.021")
e para as 4 cargas do motor (0, 1, 2, 3 HP).

Fonte oficial: https://engineering.case.edu/bearingdatacenter/download-data-file
"""

import http.client
import os
import time
import urllib.error
import urllib.request

MAX_RETRIES = 5
CHUNK_SIZE = 8192

# Pasta onde os arquivos brutos serão salvos
OUTPUT_DIR = os.path.join("data", "raw")

BASE_URL = "https://engineering.case.edu/sites/default/files/{}.mat"

# Mapeamento: nome_do_arquivo -> ID numérico no servidor da CWRU
# Cargas (HP): 0, 1, 2, 3  |  Severidades: 007, 014, 021
FILES = {
    # --- Baseline (saudável) ---
    "Normal_0.mat": 97,
    "Normal_1.mat": 98,
    "Normal_2.mat": 99,
    "Normal_3.mat": 100,

    # --- Falha na pista interna (Inner Race) ---
    "IR007_0.mat": 105, "IR007_1.mat": 106, "IR007_2.mat": 107, "IR007_3.mat": 108,
    "IR014_0.mat": 169, "IR014_1.mat": 170, "IR014_2.mat": 171, "IR014_3.mat": 172,
    "IR021_0.mat": 209, "IR021_1.mat": 210, "IR021_2.mat": 211, "IR021_3.mat": 212,

    # --- Falha em elemento rolante (Ball) ---
    "B007_0.mat": 118, "B007_1.mat": 119, "B007_2.mat": 120, "B007_3.mat": 121,
    "B014_0.mat": 185, "B014_1.mat": 186, "B014_2.mat": 187, "B014_3.mat": 188,
    "B021_0.mat": 222, "B021_1.mat": 223, "B021_2.mat": 224, "B021_3.mat": 225,

    # --- Falha na pista externa, posição centrada @6:00 (Outer Race) ---
    "OR007@6_0.mat": 130, "OR007@6_1.mat": 131, "OR007@6_2.mat": 132, "OR007@6_3.mat": 133,
    "OR014@6_0.mat": 197, "OR014@6_1.mat": 198, "OR014@6_2.mat": 199, "OR014@6_3.mat": 200,
    "OR021@6_0.mat": 234, "OR021@6_1.mat": 235, "OR021@6_2.mat": 236, "OR021@6_3.mat": 237,
}


def download_file(filename: str, file_id: int, dest_dir: str) -> None:
    """Baixa um único arquivo .mat, com retry automático em caso de conexão
    interrompida (IncompleteRead), pulando se o arquivo já existir."""
    dest_path = os.path.join(dest_dir, filename)
    tmp_path = dest_path + ".part"

    if os.path.exists(dest_path):
        print(f"[OK]   {filename} já existe, pulando.")
        return

    url = BASE_URL.format(file_id)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                expected_size = response.getheader("Content-Length")
                expected_size = int(expected_size) if expected_size else None

                with open(tmp_path, "wb") as out_file:
                    downloaded = 0
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded += len(chunk)

                # Se o servidor informou o tamanho, valida se bateu certinho
                if expected_size is not None and downloaded < expected_size:
                    raise http.client.IncompleteRead(downloaded, expected_size - downloaded)

            os.replace(tmp_path, dest_path)
            size_kb = os.path.getsize(dest_path) / 1024
            print(f"[BAIXADO] {filename} ({size_kb:.1f} KB)")
            return

        except (http.client.IncompleteRead, urllib.error.URLError, ConnectionError, TimeoutError) as e:
            print(f"[TENTATIVA {attempt}/{MAX_RETRIES}] Falha em {filename}: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            time.sleep(2 * attempt)  # espera crescente entre tentativas

    print(f"[ERRO] Não foi possível baixar {filename} após {MAX_RETRIES} tentativas.")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Salvando arquivos em: {os.path.abspath(OUTPUT_DIR)}\n")

    total = len(FILES)
    for i, (filename, file_id) in enumerate(FILES.items(), start=1):
        print(f"({i}/{total}) ", end="")
        download_file(filename, file_id, OUTPUT_DIR)
        time.sleep(0.5)  # pequena pausa para não sobrecarregar o servidor

    print("\nDownload concluído.")


if __name__ == "__main__":
    main()