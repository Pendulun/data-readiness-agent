from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import MessagesState
import pandas as pd

from src_data_readyness_agent.common import middleware
from src_data_readyness_agent.common import data_structs
from src_data_readyness_agent.data_evaluation_agent import tools


class State(MessagesState):
    dataset: pd.DataFrame
    dataset_profile: data_structs.DatasetProfile | None = None


def get_agent(openai_api_key: str,
              qt_maxima_iteracoes_agente: int,
              model: str = 'gpt-5-nano') -> CompiledStateGraph:
    """
    Retorna a instância do agente de avaliação
    """
    model = ChatOpenAI(model=model, temperature=0.0, api_key=openai_api_key)
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
    return create_agent(model=model,
                        system_prompt=system_prompt,
                        response_format=ToolStrategy(
                            data_structs.EvalAgentResponse),
                        state_schema=State,
                        tools=[
                            tools.get_columns_names,
                            tools.check_duplicate_rows,
                            tools.check_duplicate_rows_all_cols,
                            tools.check_column_consistency,
                            tools.get_column_value_distribution,
                            tools.analyze_missingness_patterns,
                            tools.detect_outliers,
                            tools.get_n_rows,
                            tools.get_n_cols,
                            tools.get_null_counts,
                            tools.get_unique_counts,
                            tools.get_col_preview,
                        ],
                        middleware=[
                            middleware.DebugMiddleware(),
                            middleware.IterationLimitMiddleware(
                                max_iterations=qt_maxima_iteracoes_agente)
                        ])
