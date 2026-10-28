"""Treina o modelo de aprovação de crédito e salva em modelos/.

Para rodar (a partir da pasta ServindoModelo/):
    python src/treinamento.py

O arquivo salvo contém o pipeline completo (pré-processamento + modelo),
então a API recebe os dados crus e não precisa repetir nenhuma transformação.
"""

from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
CAMINHO_DADOS = RAIZ_PROJETO / "data" / "credito.csv"
CAMINHO_MODELO = RAIZ_PROJETO / "modelos" / "modelo_credito.pkl"
VERSAO = "1.0.0"
SEED = 42

# Atributos sensíveis: usá-los pode levar a discriminação por gênero ou etnia.
COLUNAS_SENSIVEIS = ["Gender", "Ethnicity"]

ALVO = "Approved"
FEATURES_NUMERICAS = ["Age", "Debt", "YearsEmployed", "CreditScore", "Income"]
FEATURES_BINARIAS = ["Married", "BankCustomer", "PriorDefault", "Employed", "DriversLicense"]
FEATURES_CATEGORICAS = ["Industry", "Citizen"]
# ZipCode fica de fora: pouco poder preditivo e pode ser proxy de atributos sensíveis.
FEATURES = FEATURES_NUMERICAS + FEATURES_BINARIAS + FEATURES_CATEGORICAS


def carregar_dados() -> pd.DataFrame:
    # ZipCode é lido como texto para não perder os zeros à esquerda
    df = pd.read_csv(CAMINHO_DADOS, dtype={"ZipCode": str})
    return df.drop(columns=COLUNAS_SENSIVEIS)


def criar_pipeline(estimador) -> Pipeline:
    preprocessamento = ColumnTransformer([
        ("num", StandardScaler(), FEATURES_NUMERICAS),
        # handle_unknown="ignore" evita erro se chegar uma categoria nova em produção
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CATEGORICAS),
        ("bin", "passthrough", FEATURES_BINARIAS),
    ])
    return Pipeline([("preprocessamento", preprocessamento), ("modelo", estimador)])


def escolher_modelo(X_treino, y_treino) -> tuple[str, Pipeline]:
    """Compara os candidatos com validação cruzada e devolve o melhor."""
    candidatos = {
        "regressao_logistica": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    }
    resultados = {}
    for nome, estimador in candidatos.items():
        scores = cross_val_score(criar_pipeline(estimador), X_treino, y_treino, cv=5, scoring="roc_auc")
        resultados[nome] = scores.mean()
        print(f"{nome:22s} ROC AUC = {scores.mean():.3f} ± {scores.std():.3f}")

    melhor = max(resultados, key=resultados.get)
    print(f"\nMelhor modelo: {melhor}\n")
    return melhor, criar_pipeline(candidatos[melhor])


def avaliar(pipeline: Pipeline, X_teste, y_teste) -> None:
    y_pred = pipeline.predict(X_teste)
    y_proba = pipeline.predict_proba(X_teste)[:, 1]
    print(classification_report(y_teste, y_pred, target_names=["negado", "aprovado"]))
    print(f"ROC AUC no teste: {roc_auc_score(y_teste, y_proba):.3f}")
    print("Matriz de confusão (linhas = real, colunas = previsto):")
    print(confusion_matrix(y_teste, y_pred))


def salvar(pipeline: Pipeline, nome: str) -> None:
    # A API lê nome, versão e features daqui, sem duplicar essa informação no código.
    artefato = {
        "pipeline": pipeline,
        "nome": nome,
        "versao": VERSAO,
        "features": FEATURES,
        "sklearn_version": sklearn.__version__,
    }
    CAMINHO_MODELO.parent.mkdir(exist_ok=True)
    joblib.dump(artefato, CAMINHO_MODELO)
    print(f"\nModelo salvo em {CAMINHO_MODELO}")


def main() -> None:
    df = carregar_dados()
    X, y = df[FEATURES], df[ALVO]
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"Treino: {len(X_treino)} linhas | Teste: {len(X_teste)} linhas\n")

    nome, pipeline = escolher_modelo(X_treino, y_treino)
    pipeline.fit(X_treino, y_treino)
    avaliar(pipeline, X_teste, y_teste)
    salvar(pipeline, nome)


if __name__ == "__main__":
    main()