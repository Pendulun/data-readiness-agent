from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import MessagesState
import pandas as pd
from typing_extensions import Any, Dict, List

from data_readyness_agent import common_middleware, common_tools, data_transformation_tools


class State(MessagesState):
    dataset: pd.DataFrame
    tool_history: Dict[str, List[Dict[str, Any]]]


def get_agent(openai_api_key: str,
              qt_max_iteracoes_agente: int) -> CompiledStateGraph:
    """
    Retorna a instância do agente de transformação da base
    """
    model = ChatOpenAI(model='gpt-5-nano',
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
            common_tools.dummy_tool,
            data_transformation_tools.get_n_cols,
            data_transformation_tools.get_cols_names,
            data_transformation_tools.has_nulls,
            data_transformation_tools.get_unique_counts,
            data_transformation_tools.get_column_type,
            data_transformation_tools.get_col_unique,
            data_transformation_tools.replace_substring,
            data_transformation_tools.drop_column,
            data_transformation_tools.map_col_values,
            data_transformation_tools.copy_column,
            data_transformation_tools.convert_column_to_datetime,
            data_transformation_tools.convert_column_to_float,
            data_transformation_tools.convert_column_to_int,
            data_transformation_tools.fill_col_na,
            data_transformation_tools.rename_column,
            data_transformation_tools.subtract_value,
            data_transformation_tools.sum_value,
            data_transformation_tools.divide_value,
            data_transformation_tools.multiply_value,
            data_transformation_tools.power_value,
            data_transformation_tools.subtract_cols,
            data_transformation_tools.sum_cols,
            data_transformation_tools.divide_cols,
            data_transformation_tools.multiply_cols,
            data_transformation_tools.power_cols,
        ],
        middleware=[
            common_middleware.DebugMiddleware(),
            common_middleware.IterationLimitMiddleware(
                max_iterations=qt_max_iteracoes_agente, dummy_tool=True)
        ])
