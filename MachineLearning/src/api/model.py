"""Carregamento do modelo e geração de predições."""

import os
from pathlib import Path

import joblib
import pandas as pd

# Raiz do projeto: src/api/model.py -> sobe dois níveis até ServindoModelo/
RAIZ_PROJETO = Path(__file__).resolve().parents[2]
CAMINHO_PADRAO = RAIZ_PROJETO / "modelos" / "modelo_credito.pkl"


def carregar_modelo(caminho: str | Path | None = None) -> dict:
    """Lê o arquivo salvo pelo notebook de treinamento.

    O arquivo contém um dicionário com o pipeline (pré-processamento +
    modelo) e alguns metadados, como nome e versão.

    Atenção: pickle/joblib executa código ao carregar. Nunca carregue
    arquivos de origem desconhecida.
    """
    caminho = Path(caminho or os.getenv("MODEL_PATH", CAMINHO_PADRAO))
    if not caminho.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {caminho}. "
            "Rode o notebook notebooks/treinamento.ipynb para gerá-lo."
        )
    return joblib.load(caminho)


def prever(artefato: dict, dados: dict) -> dict:
    """Recebe os dados de UM cliente e devolve a predição."""
    pipeline = artefato["pipeline"]

    # O pipeline foi treinado com um DataFrame, então montamos um
    # DataFrame de uma linha com as colunas na mesma ordem do treino.
    entrada = pd.DataFrame([dados], columns=artefato["features"])

    probabilidade = float(pipeline.predict_proba(entrada)[0, 1])
    return {
        "aprovado": probabilidade >= 0.5,
        "probabilidade_aprovacao": round(probabilidade, 4),
        "modelo": artefato["nome"],
        "versao": artefato["versao"],
    }