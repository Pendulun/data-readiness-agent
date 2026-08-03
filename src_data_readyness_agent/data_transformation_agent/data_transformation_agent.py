from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import MessagesState
import pandas as pd

from src_data_readyness_agent.common.tools import dummy_tool
from src_data_readyness_agent.common import middleware as common_middleware
from src_data_readyness_agent.data_transformation_agent import data_structs, middleware, tools


class State(MessagesState):
    dataset: pd.DataFrame
    tool_history: data_structs.ToolHistory
    frozen_columns: set[str]  # Lista de colunas que não podem ser alteradas


def get_agent(openai_api_key: str,
              qt_max_iteracoes_agente: int,
              model: str = 'gpt-5-nano') -> CompiledStateGraph:
    """
    Retorna a instância do agente de transformação da base
    """
    model = ChatOpenAI(model=model,
                       temperature=0.0,
                       api_key=openai_api_key,
                       model_kwargs={"parallel_tool_calls": False})
    system_prompt = """
    Você é responsável por preprocessar uma base de dados para um projeto de Data Science.

    Comece utilizando o DatasetProfile dos dados originais fornecido. 
    Depois, planeje as transformações que você vai fazer seguindo a mensagem do usuário. 
    Uma vez planejado, siga o planejamento usando as ferramentas existentes para transformar os dados.
    Só faça as transformações indicadas pelo usuário.

    Evite repetir chamadas de ferramentas com os mesmos argumentos,
    a menos que exista uma justificativa clara para obter novos dados.

    Dadas as recomendações de transformações, só tente realizar aquelas possíveis de acordo
    com as ferramentas disponíveis.

    Execute apenas uma transformação por vez.

    Uma vez satisfeito, pare e não pergunte mais nada para o usuário.
    """
    return create_agent(
        model=model,
        system_prompt=system_prompt,
        state_schema=State,
        tools=[
            dummy_tool,
            tools.get_n_cols,
            tools.get_cols_names,
            tools.has_nulls,
            tools.get_unique_counts,
            tools.get_column_type,
            tools.get_col_unique,
            tools.replace_substring,
            tools.drop_column,
            tools.map_col_values,
            tools.copy_column,
            tools.convert_column_to_datetime,
            tools.convert_column_to_float,
            tools.convert_column_to_int,
            tools.fill_col_na,
            tools.rename_column,
            tools.subtract_value,
            tools.sum_value,
            tools.divide_value,
            tools.multiply_value,
            tools.power_value,
            tools.subtract_cols,
            tools.sum_cols,
            tools.divide_cols,
            tools.multiply_cols,
            tools.power_cols,
        ],
        middleware=[
            common_middleware.DebugMiddleware(),
            common_middleware.IterationLimitMiddleware(
                max_iterations=qt_max_iteracoes_agente, dummy_tool=True),
            # A ordem desses dois middlewares importa já que ambos são wrap_tool_call
            middleware.handle_tool_errors,
            middleware.add_to_tool_history,
        ])
