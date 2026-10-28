"""Contratos de entrada e saída da API.

O Pydantic valida cada requisição antes de ela chegar ao modelo:
campos faltando, tipos errados ou valores fora dos limites geram
automaticamente uma resposta 422 com a explicação do erro.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Industria = Literal[
    "CommunicationServices",
    "ConsumerDiscretionary",
    "ConsumerStaples",
    "Education",
    "Energy",
    "Financials",
    "Healthcare",
    "Industrials",
    "InformationTechnology",
    "Materials",
    "Real Estate",
    "Research",
    "Transport",
    "Utilities",
]

Cidadania = Literal["ByBirth", "ByOtherMeans", "Temporary"]

Binario = Literal[0, 1]


class DadosCliente(BaseModel):
    """Dados de um cliente. Os nomes seguem as colunas do dataset de treino."""

    Age: float = Field(ge=18, le=100, description="Idade em anos")
    Debt: float = Field(ge=0, description="Dívida atual")
    Married: Binario = Field(description="Casado(a): 1 = sim, 0 = não")
    BankCustomer: Binario = Field(description="Já é cliente do banco")
    Industry: Industria = Field(description="Setor em que trabalha")
    YearsEmployed: float = Field(ge=0, description="Anos de emprego")
    PriorDefault: Binario = Field(description="Indicador de inadimplência anterior (codificado como no dataset)")
    Employed: Binario = Field(description="Está empregado(a)")
    CreditScore: int = Field(ge=0, description="Pontuação de crédito")
    DriversLicense: Binario = Field(description="Possui CNH")
    Citizen: Cidadania = Field(description="Tipo de cidadania")
    Income: float = Field(ge=0, description="Renda")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "Age": 30.83,
                    "Debt": 0.0,
                    "Married": 1,
                    "BankCustomer": 1,
                    "Industry": "Industrials",
                    "YearsEmployed": 1.25,
                    "PriorDefault": 1,
                    "Employed": 1,
                    "CreditScore": 1,
                    "DriversLicense": 0,
                    "Citizen": "ByBirth",
                    "Income": 0,
                }
            ]
        }
    )


class Predicao(BaseModel):
    aprovado: bool
    probabilidade_aprovacao: float
    modelo: str
    versao: str