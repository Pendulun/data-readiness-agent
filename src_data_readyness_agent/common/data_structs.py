import copy
from enum import Enum
from pydantic import BaseModel, Field
from typing_extensions import Any, Dict, List


class ReadinessStatus(str, Enum):
    READY = "ready"
    READY_WITH_ISSUES = "ready_with_issues"
    NOT_READY = "not_ready"


class Finding(BaseModel):
    column: str | None = Field(default=None, description="O nome da coluna")
    category: str = Field(description="A categoria da coluna")
    severity: str = Field(description="O nível de problema nessa coluna")
    description: str = Field(description="Uma descrição para a coluna")
    recommendation: str = Field(
        description="Uma recomendação de ação concisa final para a coluna")


class EvalAgentResponse(BaseModel):
    readiness_status: ReadinessStatus = Field(
        description="O nível de preparo geral da base")
    summary: str = Field(description="Um resumo dos achados sobre a base")
    findings: list[Finding] = Field(
        description="Uma lista de achados por coluna")

    def to_markdown(self) -> str:
        output = [
            f"### Status: {str(self.readiness_status.value).title()}", "",
            f"### Resumo: \n{self.summary}", "", "### Problemas encontrados:"
        ]

        output.append(self.get_findings_str())

        return "\n".join(output)

    def get_findings_str(self) -> str:
        output = list()
        for finding in self.findings:
            output.append(
                f"- [{finding.severity.upper()}] "
                f"{finding.column}: {finding.description} {finding.recommendation}"
            )
        return "\n".join(output)


class DatasetProfile(BaseModel):
    n_rows: int = Field(description="Quantidade de linhas na base")
    n_columns: int = Field(description="Quantidade de colunas na base")
    columns_types: dict[str, str] = Field(description="Tipos das colunas")
    null_counts: dict[str, int] = Field(
        description="Quantidade de nulos por coluna")
    unique_counts: dict[str, int] = Field(
        description="Quantidade de valores únicos por coluna")
    samples: dict[str, list] = Field(description="Amostra de dados por coluna")
