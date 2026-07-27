from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import MessagesState
import pandas as pd
from pydantic import BaseModel, Field
from typing_extensions import Any, Dict, List

from data_readyness_agent import common_middleware, data_transformation_tools


class AgentResponse(BaseModel):
    status: str = Field(
        description=
        "Um status representando se as transformações foram aplicadas ou não")
    # transformations: List[str] = Field(
    #     description="Uma lista de transformações realizadas na base")

    # def to_markdown(self) -> str:
    #     output = ["### Transformações realizadas na base:"]

    #     for val in self.transformations:
    #         output.append(f"- {val} ")

    #     return "\n".join(output)


class State(MessagesState):
    dataset: pd.DataFrame
    tool_history: Dict[str, List[Dict[str, Any]]]


def get_agent(openai_api_key: str) -> CompiledStateGraph:
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
        #response_format=ToolStrategy(AgentResponse),
        state_schema=State,
        tools=[
            data_transformation_tools.get_n_cols,
            data_transformation_tools.get_cols_names,
            data_transformation_tools.has_nulls,
            data_transformation_tools.get_unique_counts,
            data_transformation_tools.get_column_type,
            data_transformation_tools.get_col_unique_preview,
            data_transformation_tools.replace_substring,
            data_transformation_tools.drop_column,
            data_transformation_tools.convert_column_to_datetime,
            data_transformation_tools.convert_column_to_float,
            data_transformation_tools.convert_column_to_int,
            data_transformation_tools.fill_col_na,
        ],
        middleware=[common_middleware.DebugMiddleware()])
