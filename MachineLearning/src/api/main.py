"""API de aprovação de crédito.

Para rodar localmente (a partir da pasta ServindoModelo/):
    uvicorn src.api.main:app --reload

Documentação interativa: http://localhost:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.model import carregar_modelo, prever
from src.api.schemas import DadosCliente, Predicao

artefato = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carrega o modelo UMA vez, quando o servidor inicia,
    # e não a cada requisição (seria muito lento).
    artefato.update(carregar_modelo())
    yield
    artefato.clear()


app = FastAPI(
    title="API de Crédito",
    description="Exemplo de como servir um modelo de Machine Learning com FastAPI.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Verifica se a API está no ar e se o modelo foi carregado."""
    return {"status": "ok", "modelo": artefato.get("nome"), "versao": artefato.get("versao")}


@app.post("/predict/", response_model=Predicao)
def predict(cliente: DadosCliente):
    """Recebe os dados de um cliente e retorna se o crédito seria aprovado."""
    return prever(artefato, cliente.model_dump())