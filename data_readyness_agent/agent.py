from enum import Enum
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import tool, ToolRuntime
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import MessagesState
import pandas as pd
from pydantic import BaseModel, Field
from typing_extensions import Any, Dict, List


class DebugMiddleware(AgentMiddleware):

    def after_model(self, state, runtime):
        last_message = state["messages"][-1]

        if isinstance(last_message, AIMessage):
            print("\n--- LLM RESPONSE ---")
            print("Content:", last_message.content)
            print("Tool calls:", last_message.tool_calls)

        return None


class IterationLimitMiddleware(AgentMiddleware):

    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations

    def wrap_model_call(self, request, handler):
        iteration = self.count_model_calls(request.state["messages"])

        print(f"Iteração do agente: "
              f"{iteration}/{self.max_iterations}")

        # A última iteração é reservada para gerar a resposta final
        if iteration >= self.max_iterations - 1:
            print("Limite de iterações atingido.")

            request = request.override(
                tools=[],
                messages=[
                    *request.messages,
                    SystemMessage(
                        content=("O limite de investigação foi atingido. "
                                 "Não execute mais ferramentas. "
                                 "Gere agora a resposta final estruturada "
                                 "com base nas informações coletadas."))
                ])

        return handler(request)

    def count_model_calls(self, messages) -> int:
        # Conta quantas respostas da LLM já existem
        return sum(1 for message in messages
                   if message.__class__.__name__ == "AIMessage")


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


@tool
def get_n_rows(runtime: ToolRuntime) -> int:
    """Retorna a quantidade de linhas na base"""
    return runtime.state['dataset_profile'].n_rows


@tool
def get_n_cols(runtime: ToolRuntime) -> int:
    """Retorna a quantidade de colunas na base"""
    return runtime.state['dataset_profile'].n_columns


@tool
def get_null_counts(runtime: ToolRuntime) -> int:
    """Retorna a quantidade de valores nulos de cada coluna na base"""
    return runtime.state['dataset_profile'].null_counts


@tool
def get_unique_counts(runtime: ToolRuntime) -> int:
    """Retorna a quantidade de valores únicos em cada coluna na base"""
    return runtime.state['dataset_profile'].unique_counts


@tool
def get_col_preview(column: str,
                    runtime: ToolRuntime) -> Dict[str, List[Any] | str]:
    """Retorna um preview dos valores contidos na coluna da base"""
    data_profile = runtime.state['dataset_profile']
    if column not in data_profile.samples.keys():
        return {"error": f"Coluna {column} não está presente na base"}

    return data_profile.samples[column]


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

    qt_duplicados = int(df.duplicated(subset=target_cols).sum())
    return {'qt_duplicados': qt_duplicados}


# Essa função existe para evitar de o agente chamar a função check_duplicate_rows
# informando todas as colunas existentes
@tool
def check_duplicate_rows_all_cols(runtime: ToolRuntime) -> Dict[str, int]:
    """Retorna a quantidade de linhas duplicadas usando todas as colunas da base"""
    df = pd.read_csv(runtime.state['data_url'])

    qt_duplicados = int(df.duplicated().sum())
    return {'qt_duplicados': qt_duplicados}


@tool
def check_column_consistency(col_name: str, runtime: ToolRuntime) -> dict:
    """Retorna quantos tipos de dados diferentes a coluna informada possui"""
    df = pd.read_csv(runtime.state["data_url"])

    if col_name not in df.columns:
        return {"error": f"A coluna '{col_name}' não existe na base."}

    # Tipos Python encontrados
    value_types = (df[col_name].dropna().map(
        lambda x: type(x).__name__).value_counts().to_dict())

    return value_types


@tool
def get_column_value_distribution(col_name: str, runtime: ToolRuntime) -> dict:
    """Retorna a distribuição de até 50 valores mais comuns da coluna informada"""
    df = pd.read_csv(runtime.state["data_url"])

    if col_name not in df.columns:
        return {"error": f"A coluna '{col_name}' não existe."}

    counts = (df[col_name].value_counts(dropna=False).head(50).to_dict())

    return {str(value): int(count) for value, count in counts.items()}


@tool
def analyze_missingness_patterns(col_name: str, runtime: ToolRuntime) -> dict:
    """
    Analisa se valores ausentes da coluna informada estão associados a outras colunas categóricas.
    Só chame para colunas que, de fato, possuam valores nulos.
    """
    df = pd.read_csv(runtime.state["data_url"])

    if col_name not in df.columns:
        return {"error": f"Coluna '{col_name}' não encontrada."}

    missing_mask = df[col_name].isna()

    if missing_mask.sum() == 0:
        return {"msg": "A coluna não possui valores faltantes."}

    results = {}

    for col in df.columns:
        if col == col_name:
            continue

        # Se for uma coluna categórica
        if df[col].dtype == "object":
            # Conta a porcentagem de valores em col em que col_name é nulo
            grouped = (missing_mask.groupby(
                df[col]).mean().sort_values(ascending=False))

            results[col] = {
                str(k): round(float(v), 2)
                for k, v in grouped.items()
            }

    return results


@tool
def detect_outliers(col_name: str, runtime: ToolRuntime) -> dict:
    """
    Detecta possíveis outliers na coluna numérica informada usando o método IQR.

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
        "q1":
        round(float(q1), 2),
        "q3":
        round(float(q3), 2),
        "iqr":
        round(float(iqr), 2),
        "lower_bound":
        round(float(lower_bound), 2),
        "upper_bound":
        round(float(upper_bound), 2),
        "n_outliers":
        n_outliers,
        "outlier_percentage":
        round((n_outliers / n_valid * 100 if n_valid > 0 else 0), 2)
    }


@tool
def get_columns_names(runtime: ToolRuntime) -> List[str]:
    """Retorna os nomes de todas as colunas existentes na base"""
    df = pd.read_csv(runtime.state['data_url'])
    return df.columns.tolist()


def get_agent(openai_api_key: str,
              qt_maxima_iteracoes_agente: int = 15) -> CompiledStateGraph:
    model = ChatOpenAI(model='gpt-5-nano',
                       temperature=0.0,
                       api_key=openai_api_key)
    system_prompt = """
    Você é responsável por avaliar a prontidão de uma base de dados
    para um projeto de Data Science.

    Comece utilizando o DatasetProfile fornecido. Depois, planeje a investigação
    que você vai fazer. Uma vez planejado, siga o planejamento usando as ferramentas
    necessárias.

    Não solicite novamente informações já disponíveis no DatasetProfile.

    Use as ferramentas apenas quando:
    1. uma informação não estiver disponível no perfil;
    2. for necessário aprofundar uma possível inconsistência;
    3. for necessário validar uma hipótese levantada durante a análise.

    Evite repetir chamadas de ferramentas com os mesmos argumentos,
    a menos que exista uma justificativa clara para obter novos dados.
    """
    return create_agent(
        model=model,
        system_prompt=system_prompt,
        response_format=ToolStrategy(AgentResponse),
        state_schema=State,
        tools=[
            get_columns_names,
            check_duplicate_rows,
            check_duplicate_rows_all_cols,
            check_column_consistency,
            get_column_value_distribution,
            analyze_missingness_patterns,
            detect_outliers,
            get_n_rows,
            get_n_cols,
            get_null_counts,
            get_unique_counts,
            get_col_preview,
        ],
        middleware=[
            DebugMiddleware(),
            IterationLimitMiddleware(max_iterations=qt_maxima_iteracoes_agente)
        ])


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


def get_avaliacao(data_url: str, openai_api_key: str,
                  qt_maxima_iteracoes_agente: int) -> AgentResponse:
    agent = get_agent(openai_api_key, qt_maxima_iteracoes_agente)
    profile = create_dataset_profile(data_url)
    profile_text = profile.model_dump_json(indent=2)

    response = agent.invoke({
        'messages': [
            HumanMessage(content=f"""
                Avalie essa base de dados.

                Você já possui o seguinte perfil inicial da base:

                {profile_text}

                Use essas informações como ponto de partida.
                Não repita análises que já estão presentes no perfil.
                Use as ferramentas disponíveis apenas para aprofundar
                a investigação de possíveis problemas de qualidade.
                """),
        ],
        'data_url':
        data_url,
        "dataset_profile":
        profile,
    })
    final_response: AgentResponse = response['structured_response']
    return final_response
