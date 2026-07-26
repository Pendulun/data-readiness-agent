from enum import Enum
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.agents.middleware import AgentMiddleware
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import MessagesState
import pandas as pd
from pydantic import BaseModel, Field

from data_readyness_agent import data_evaluation_agent


def create_dataset_profile(
        df: pd.DataFrame) -> data_evaluation_agent.DatasetProfile:

    return data_evaluation_agent.DatasetProfile(
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
                  qt_maxima_iteracoes_agente: int,
                  target_col: str) -> data_evaluation_agent.AgentResponse:
    dataset = pd.read_csv(data_url)
    agent = data_evaluation_agent.get_agent(openai_api_key,
                                            qt_maxima_iteracoes_agente)
    profile = create_dataset_profile(dataset)
    profile_text = profile.model_dump_json(indent=2)

    response = agent.invoke({
        'messages': [
            HumanMessage(content=f"""
                Avalie essa base de dados.

                Você já possui o seguinte perfil inicial da base:

                {profile_text}

                A coluna alvo do modelo é {target_col}.
                
                Use essas informações como ponto de partida.
                Não repita análises que já estão presentes no perfil.
                Use as ferramentas disponíveis apenas para aprofundar
                a investigação de possíveis problemas de qualidade.
                """),
        ],
        'dataset':
        dataset,
        "dataset_profile":
        profile,
    })
    final_response: data_evaluation_agent.AgentResponse = response[
        'structured_response']
    return final_response
