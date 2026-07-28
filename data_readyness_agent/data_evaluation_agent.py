from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.agents.middleware import AgentMiddleware
from langchain.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import MessagesState
import pandas as pd

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
        response_format=ToolStrategy(common_data_structs.EvalAgentResponse),
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
