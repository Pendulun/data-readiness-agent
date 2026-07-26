from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import MessagesState
import pandas as pd
from pydantic import BaseModel, Field
from typing_extensions import List

from data_readyness_agent import common_data_structs, common_middleware, common_tools, data_transformation_tools


class AgentResponse(BaseModel):
    transformations: List[str] = Field(
        description="Uma lista de transformações realizadas na base")

    def to_markdown(self) -> str:
        output = ["### Transformações realizadas na base:"]

        for val in self.transformations:
            output.append(f"- {val} ")

        return "\n".join(output)


class State(MessagesState):
    dataset: pd.DataFrame
    original_dataset: pd.DataFrame
    dataset_profile: common_data_structs.DatasetProfile | None = None


def get_agent(openai_api_key: str) -> CompiledStateGraph:
    model = ChatOpenAI(model='gpt-5-nano',
                       temperature=0.0,
                       api_key=openai_api_key,
                       model_kwargs={"parallel_tool_calls": False})
    system_prompt = """
    Você é responsável por transformar uma base de dados
    para um projeto de Data Science.

    Comece utilizando o DatasetProfile dos dados originais fornecido. Depois, planeje as transformações
    que você vai fazer de acordo com a mensagem do usuário. Uma vez planejado, siga o planejamento usando
    as ferramentas existentes para transformar os dados.

    Evite repetir chamadas de ferramentas com os mesmos argumentos,
    a menos que exista uma justificativa clara para obter novos dados.

    Execute apenas uma transformação por vez.

    Após executar uma transformação, aguarde o resultado.
    Analise o estado atualizado da base antes de decidir a próxima transformação.

    Não execute múltiplas transformações simultaneamente.
    """
    return create_agent(
        model=model,
        system_prompt=system_prompt,
        response_format=ToolStrategy(AgentResponse),
        state_schema=State,
        tools=[
            # common_tools.get_n_rows,
            # common_tools.get_n_cols,
            # common_tools.get_null_counts,
            # common_tools.get_unique_counts,
            # common_tools.get_col_preview,
            data_transformation_tools.convert_column_to_datetime,
            data_transformation_tools.convert_column_to_float,
            data_transformation_tools.convert_column_to_int,
            data_transformation_tools.fill_col_na,
        ],
        middleware=[common_middleware.DebugMiddleware()])
