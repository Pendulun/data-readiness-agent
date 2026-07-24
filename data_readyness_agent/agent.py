from enum import Enum
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import tool, ToolRuntime
from langchain.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import MessagesState
import pandas as pd
from pprint import pprint
from pydantic import BaseModel, Field
from typing_extensions import Dict, List


class DebugMiddleware(AgentMiddleware):

    def after_model(self, state, runtime):
        last_message = state["messages"][-1]

        if isinstance(last_message, AIMessage):
            print("\n--- LLM RESPONSE ---")
            print("Content:", last_message.content)
            print("Tool calls:", last_message.tool_calls)

        return None


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
        description="Uma recomendação final para a coluna")


class AgentResponse(BaseModel):
    readiness_status: ReadinessStatus = Field(
        description="O nível de preparo geral da base")
    summary: str = Field(description="Um resumo dos achados sobre a base")
    findings: list[Finding] = Field(
        description="Uma lista de achados por coluna")
    recommended_actions: list[str] = Field(
        description="Uma lista de ações a serem tomadas para melhorar a base")

    def to_markdown(self) -> str:
        output = [
            f"### Status: \n{self.readiness_status}", "",
            f"### Resumo: \n{self.summary}", "", "### Problemas encontrados:"
        ]

        for finding in self.findings:
            output.append(f"- [{finding.severity.upper()}] "
                          f"{finding.column}: {finding.description}")

        output.append("")
        output.append("### Ações recomendadas:")

        for i, action in enumerate(self.recommended_actions, 1):
            output.append(f"{i}. {action}")

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


class State(MessagesState):
    data_url: str
    dataset_profile: DatasetProfile | None = None
    findings: list[str] = []


@tool
def check_duplicate_rows(columns: List[str],
                         runtime: ToolRuntime) -> Dict[str, int]:
    """Retorna a quantidade de linhas duplicadas usando o subconjunto de colunas informadas"""
    df = pd.read_csv(runtime.state['data_url'])
    target_cols = list(set([col for col in columns if col in df.columns]))
    if not target_cols:
        return {
            "error": ("Nenhuma das colunas informadas existe na base.",
                      "Colunas solicitadas: " + str(columns),
                      "Colunas disponíveis: " + str(df.columns.tolist()))
        }

    qt_duplicados = df.duplicated(subset=target_cols).sum()
    return {'qt_duplicados': qt_duplicados}


@tool
def check_column_consistency(col_name: str, runtime: ToolRuntime) -> dict:
    """Analisa os tipos nos valores de uma coluna"""
    df = pd.read_csv(runtime.state["data_url"])

    if col_name not in df.columns:
        return {"error": f"A coluna '{col_name}' não existe na base."}

    # Tipos Python encontrados
    value_types = (df[col_name].dropna().map(
        lambda x: type(x).__name__).value_counts().to_dict())

    return value_types


@tool
def get_column_value_distribution(col_name: str, runtime: ToolRuntime) -> dict:
    """Retorna a distribuição dos 50 valores mais comuns de uma coluna"""
    df = pd.read_csv(runtime.state["data_url"])

    if col_name not in df.columns:
        return {"error": f"A coluna '{col_name}' não existe."}

    counts = (df[col_name].value_counts(dropna=False).head(50).to_dict())

    return {str(value): int(count) for value, count in counts.items()}


@tool
def analyze_missingness_patterns(col_name: str, runtime: ToolRuntime) -> dict:
    """Analisa se valores ausentes de uma coluna estão associados a outras colunas categóricas"""
    df = pd.read_csv(runtime.state["data_url"])

    if col_name not in df.columns:
        return {"error": f"Coluna '{col_name}' não encontrada."}

    missing_mask = df[col_name].isna()

    results = {}

    for col in df.columns:
        if col == col_name:
            continue

        # Se for uma coluna categórica
        if df[col].dtype == "object":
            # Conta a porcentagem de valores em col em que col_name é nulo
            grouped = (missing_mask.groupby(
                df[col]).mean().sort_values(ascending=False))

            results[col] = {str(k): float(v) for k, v in grouped.items()}

    return results


@tool
def detect_outliers(col_name: str, runtime: ToolRuntime) -> dict:
    """
    Detecta possíveis outliers em uma coluna numérica usando o método IQR.

    Retorna a quantidade, o percentual de outliers e os limites
    inferior e superior para a detecção.
    """
    df = pd.read_csv(runtime.state["data_url"])

    if col_name not in df.columns:
        return {"error": f"A coluna '{col_name}' não existe na base."}

    col = df[col_name]

    # Verifica se a coluna é numérica
    if not pd.api.types.is_numeric_dtype(col):
        return {
            "message": "A coluna não é numérica. Não é possível aplicar IQR."
        }

    # Remove valores nulos para calcular os quartis
    values = col.dropna()

    if len(values) == 0:
        return {"message": "A coluna não possui valores válidos."}

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    is_outlier = ((col < lower_bound) | (col > upper_bound))

    n_outliers = int(is_outlier.sum())
    n_valid = int(values.shape[0])

    return {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "n_outliers": n_outliers,
        "outlier_percentage":
        (n_outliers / n_valid * 100 if n_valid > 0 else 0)
    }


def get_agent(openai_api_key: str) -> CompiledStateGraph:
    model = ChatOpenAI(model='gpt-5-nano',
                       temperature=0.0,
                       api_key=openai_api_key)
    system_prompt = """
    Você é um cientista de dados que está avaliando a base de dados informada. Seu objetivo é
    atestar a qualidade das colunas de acordo com a tarefa informada. Use as ferramentas disponíveis para 
    analisar os dados e montar sua resposta. Seja conciso e estruture a sua resposta 
    de acordo com o formato indicado
    """
    return create_agent(model=model,
                        system_prompt=system_prompt,
                        response_format=ToolStrategy(AgentResponse),
                        state_schema=State,
                        tools=[
                            check_duplicate_rows,
                            check_column_consistency,
                            get_column_value_distribution,
                            analyze_missingness_patterns,
                            detect_outliers,
                        ],
                        middleware=[DebugMiddleware()])


def create_dataset_profile(data_url: str) -> DatasetProfile:
    df = pd.read_csv(data_url)

    return DatasetProfile(
        n_rows=len(df),
        n_columns=len(df.columns),
        columns_types={
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },
        null_counts={
            col: int(qt)
            for col, qt in df.isna().sum().items()
        },
        unique_counts={
            col: int(qt)
            for col, qt in df.nunique().items()
        },
        samples={col: df[col].head(5).tolist()
                 for col in df.columns})


def get_avaliacao(data_url: str, openai_api_key: str) -> AgentResponse:
    agent = get_agent(openai_api_key)
    profile = create_dataset_profile(data_url)
    response = agent.invoke({
        'messages': [HumanMessage(content="Avalie essa base de dados")],
        'data_url':
        data_url,
        "dataset_profile":
        profile
    })
    pprint(response)
    final_response: AgentResponse = response['structured_response']
    return final_response.to_markdown()
