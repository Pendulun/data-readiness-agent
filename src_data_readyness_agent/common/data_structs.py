from enum import Enum
from pydantic import BaseModel, Field
from typing_extensions import List


class ReadinessStatus(str, Enum):
    READY = "ready"
    READY_WITH_ISSUES = "ready_with_issues"
    NOT_READY = "not_ready"


class Severity(str, Enum):
    VERY_HIGH = "very high"
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'
    VERY_LOW = 'low'
    NO_PROBLEM = 'no problem'


class ActionType(str, Enum):
    FILL_MISSING_VALUES = "fill_missing_values"
    DROP_COLUMN = "drop_column"
    CONVERT_DTYPE = "convert_dtype"
    ENCODE_CATEGORICAL = 'encode_categorical'
    REPLACE_SUBSTRING = "replace_substring"
    MODIFY_NUMERIC_COL = "modify_numeric_col"
    CREATE_DERIVED_COL = "create_derived_col"


class Action(BaseModel):
    explanation: str = Field(
        "Uma explicação concisa do por que a ação deve ser tomada, explicitando o problema"
    )
    recommended_action_type: ActionType = Field(
        description="O tipo da ação recomendada")

    def to_str(self):
        return self.explanation + f" Suggested action: {self.recommended_action_type.value.capitalize()}"


class ColumnIssue(BaseModel):
    column: str = Field(description="O nome da coluna analisada")
    severity: Severity = Field(
        description="A severidade geral dos problemas na coluna")
    suggested_actions: List[Action] = Field(
        description=
        "Informações sobre ações recomendadas sobre a coluna analisada")

    def to_str(self) -> str:
        txt = f"[{self.severity.value.capitalize()}] {self.column}"
        actions_txt = list()
        for action in self.suggested_actions:
            actions_txt.append(action.to_str())

        if len(actions_txt) > 0:
            txt += ": " + ". ".join(actions_txt)
        return txt


class EvalAgentResponse(BaseModel):
    readiness_status: ReadinessStatus = Field(
        description="O nível de preparo geral da base")
    summary: str = Field(description="Um resumo dos achados sobre a base")
    findings: list[ColumnIssue] = Field(
        description="Uma lista de achados por coluna")

    def to_markdown(self) -> str:
        output = [
            f"### Status: {str(self.readiness_status.value).title()}", "",
            f"### Resumo: \n{self.summary}", "", "### Problemas encontrados:"
        ]

        output.append(self.get_suggested_actions_str())

        return "\n".join(output)

    def get_suggested_actions_str(self) -> str:
        output = list()
        for finding in self.findings:
            txt = f"- {finding.to_str()}"
            output.append(txt)
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
