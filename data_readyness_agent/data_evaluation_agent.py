from enum import Enum
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.agents.middleware import AgentMiddleware
from langchain.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import MessagesState
import pandas as pd
from pydantic import BaseModel, Field

from data_readyness_agent import common_data_structs, common_middleware, common_tools, data_evaluation_tools


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
        description="Uma recomendação de ação concisa final para a coluna")


class AgentResponse(BaseModel):
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


class State(MessagesState):
    dataset: pd.DataFrame
    dataset_profile: common_data_structs.DatasetProfile | None = None


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

    Responda em português.

    Não solicite mais informações do usuário, apenas preencha a estrutura de dados da resposta
    """
    return create_agent(
        model=model,
        system_prompt=system_prompt,
        response_format=ToolStrategy(AgentResponse),
        state_schema=State,
        tools=[
            data_evaluation_tools.get_columns_names,
            data_evaluation_tools.check_duplicate_rows,
            data_evaluation_tools.check_duplicate_rows_all_cols,
            data_evaluation_tools.check_column_consistency,
            data_evaluation_tools.get_column_value_distribution,
            data_evaluation_tools.analyze_missingness_patterns,
            data_evaluation_tools.detect_outliers,
            common_tools.get_n_rows,
            common_tools.get_n_cols,
            common_tools.get_null_counts,
            common_tools.get_unique_counts,
            common_tools.get_col_preview,
        ],
        middleware=[
            common_middleware.DebugMiddleware(),
            IterationLimitMiddleware(max_iterations=qt_maxima_iteracoes_agente)
        ])
