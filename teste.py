import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv("data/processed/features.csv", dtype={"severidade": str})
colunas = ["rms", "curtose", "fator_crista"]

arquivos = df[["arquivo", "classe"]].drop_duplicates().reset_index(drop=True)
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
folds = list(skf.split(arquivos["arquivo"], arquivos["classe"]))
idx_treino, idx_teste = folds[2]

df_treino = df[df["arquivo"].isin(arquivos.loc[idx_treino, "arquivo"])]
df_teste = df[df["arquivo"].isin(arquivos.loc[idx_teste, "arquivo"])]

modelos = {
    "RF": RandomForestClassifier(random_state=42),
    "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf")),
}

for nome, modelo in modelos.items():
    modelo.fit(df_treino[colunas], df_treino["classe"])
    res = df_teste[["arquivo", "classe"]].copy()
    res["previsto"] = modelo.predict(df_teste[colunas])
    print(f"\n=== {nome} (fold 2, base_3) ===")
    print(pd.crosstab(res["arquivo"], res["previsto"]))