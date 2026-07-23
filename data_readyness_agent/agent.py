from dataclasses import dataclass
from dotenv import load_dotenv
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


class DebugMiddleware(AgentMiddleware):

    def after_model(self, state, runtime):
        last_message = state["messages"][-1]

        if isinstance(last_message, AIMessage):
            print("\n--- LLM RESPONSE ---")
            print("Content:", last_message.content)
            print("Tool calls:", last_message.tool_calls)

        return None


class AgentResponse(BaseModel):
    avaliacao_final: str = Field(
        description=
        "Avaliação final da qualidade e viabilidade da base de dados")


@dataclass
class State(MessagesState):
    data_url: str


@tool
def get_columns_types(runtime: ToolRuntime):
    """
    Retorna os nomes das colunas e os seus tipos separados por ;
    O resultado dessa função não muda entre chamadas
    """
    print("Chamou columns types")
    df = pd.read_csv(runtime.state['data_url'])
    dtypes = df.dtypes.to_dict()

    return ";".join([f'{col}:{dtype}' for col, dtype in dtypes.items()])


@tool
def get_data_shape(runtime: ToolRuntime):
    """
    Retorna o shape da base de dados. O resultado dessa função não muda entre chamadas
    """
    print("Chamou get data shape")
    df = pd.read_csv(runtime.state['data_url'])
    return f"{df.shape[0]} linhas e {df.shape[1]} colunas"


@tool
def get_col_sample(col_name: str, runtime: ToolRuntime):
    """
    Retorna uma amostra dos valores contidos em uma coluna específica separados por ;
    """
    print("Chamou col sample:", col_name)
    df = pd.read_csv(runtime.state['data_url'])
    sample = df[col_name].sample(n=min(5, df.shape[0])).to_list()
    return ";".join([str(val) for val in sample])


@tool
def get_qt_nulls(runtime: ToolRuntime):
    """
    Retorna a quantidade de nulos em todas as colunas da base separados por ;
    O resultado dessa função não muda entre chamadas
    """
    print("Chamou qt nulls")
    df = pd.read_csv(runtime.state['data_url'])
    null_count = df.isna().sum().to_dict()
    return ";".join(
        [f'{col}:{qt_nulls}' for col, qt_nulls in null_count.items()])


@tool
def get_qt_unique_values_in_col(col_name: str, runtime: ToolRuntime):
    """
    Retorna a quantidade de valores únicos na coluna informada
    """
    print("Chamou qt unique values in col:", col_name)
    df = pd.read_csv(runtime.state['data_url'])
    return str(df[col_name].nunique())


def get_agent() -> CompiledStateGraph:
    model = ChatOpenAI(model='gpt-5-nano', temperature=0.0)
    system_prompt = """
    Você é um cientista de dados que está avaliando a base de dados informada. Seu objetivo é
    atestar a qualidade das colunas de acordo com a tarefa informada. Use apenas as ferramentas disponíveis.
    Seja conciso e estruture a sua resposta final de acordo com o formato indicado
    """
    return create_agent(model=model,
                        system_prompt=system_prompt,
                        response_format=ToolStrategy(AgentResponse),
                        state_schema=State,
                        tools=[
                            get_columns_types, get_col_sample, get_qt_nulls,
                            get_qt_unique_values_in_col, get_data_shape
                        ],
                        middleware=[DebugMiddleware()])


def get_avaliacao(data_url: str) -> str:
    load_dotenv()
    agent = get_agent()
    response = agent.invoke({
        'messages': [HumanMessage(content="Avalie essa base de dados")],
        'data_url':
        data_url
    })
    pprint(response)
    return response['structured_response'].avaliacao_final
