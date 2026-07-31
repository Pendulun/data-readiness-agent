from langchain.messages import HumanMessage
import pandas as pd
from typing_extensions import Tuple

from src_data_readyness_agent.common import data_structs
from src_data_readyness_agent.data_evaluation_agent import data_evaluation_agent
from src_data_readyness_agent.data_transformation_agent import data_transformation_agent
from src_data_readyness_agent.data_transformation_agent.data_structs import ToolHistory


def get_avaliacao(dataset: pd.DataFrame, openai_api_key: str,
                  qt_maxima_iteracoes_agente: int, target_col: str,
                  qt_maxima_supersteps: int,
                  prefered_language: str) -> data_structs.EvalAgentResponse:
    """
    Gera avaliação da base
    Args:
        dataset (pd.DataFrame):
            Dataset a ser transformado
        openai_api_key (str):
            Chave da OpenAI para fazer requisições
        qt_maxima_iteracoes_agente (int):
            Quantidade máxima de iterações de avaliação do agente. Isso restringe o esforço
            e tempo necessário para que o agente analise a base e evita que ele gaste muito
            tempo e tokens na análise
        target_col (str):
            Coluna alvo da base que será usada como alvo em uma tarefa de modelagem
    """
    profile = create_dataset_profile(dataset)
    avaliacao = generate_avaliacao(
        dataset,
        profile,
        openai_api_key=openai_api_key,
        qt_maxima_iteracoes_agente=qt_maxima_iteracoes_agente,
        target_col=target_col,
        qt_maxima_supersteps=qt_maxima_supersteps,
        prefered_language=prefered_language)
    return avaliacao


def get_base_transformada(
        findings_str: str, dataset: pd.DataFrame, openai_api_key: str,
        qt_max_iteracoes_agente: int,
        qt_maxima_supersteps: int) -> Tuple[pd.DataFrame, ToolHistory]:
    """
    Aplica transformações na base

    Args:
        findings_str (str):
            Texto representando os problemas encontrados na base
        dataset (pd.DataFrame):
            Dataset a ser transformado
        openai_api_key (str):
            Chave da OpenAI para fazer requisições
        qt_max_iteracoes_agente (int):
            Quantidade máxima de iterações do agente

    Returns:
        Dataset transformado e o histórico de chamadas com sucesso a tools
        de transformação
    """
    # Conseguir o profile da base é barato então eu posso calcular aqui de novo
    profile = create_dataset_profile(dataset)
    dataset_transformado, tool_history = get_transformed_df(
        findings_str,
        profile,
        dataset,
        openai_api_key,
        qt_max_iteracoes_agente=qt_max_iteracoes_agente,
        qt_maxima_supersteps=qt_maxima_supersteps)

    return dataset_transformado, tool_history


def create_dataset_profile(df: pd.DataFrame) -> data_structs.DatasetProfile:
    """
    Cria um perfil da base
    """

    return data_structs.DatasetProfile(
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


def generate_avaliacao(
        dataset: pd.DataFrame, profile: data_structs.DatasetProfile,
        openai_api_key: str, qt_maxima_iteracoes_agente: int, target_col: str,
        qt_maxima_supersteps: int,
        prefered_language: str) -> data_structs.EvalAgentResponse:
    """
    Invoca o agente responsável por gerar a avaliação da base
    """

    agent = data_evaluation_agent.get_agent(openai_api_key,
                                            qt_maxima_iteracoes_agente)
    profile_text = profile.model_dump_json(indent=2)

    response = agent.invoke(
        {
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

                Gere a resposta em {prefered_language}
                """),
            ],
            'dataset':
            dataset,
            "dataset_profile":
            profile,
        },
        config={"recursion_limit": qt_maxima_supersteps})
    final_response: data_structs.EvalAgentResponse = response[
        'structured_response']
    return final_response


def get_transformed_df(
        avaliacao: str, profile: data_structs.DatasetProfile,
        dataset: pd.DataFrame, openai_api_key: str,
        qt_max_iteracoes_agente: int,
        qt_maxima_supersteps: int) -> Tuple[pd.DataFrame, ToolHistory]:
    """
    Invoca o agente responsável por aplicar transformações sobre o dataset
    original de acordo com a avalição feita

    Returns:
        O dataset transformado e o histórico de uso de sucesso das tools de transformação
        da base por coluna
    """
    agent = data_transformation_agent.get_agent(openai_api_key,
                                                qt_max_iteracoes_agente)
    profile_text = profile.model_dump_json(indent=2)

    response = agent.invoke(
        {
            'messages': [
                HumanMessage(content=f"""
                Transforme essa base de dados.

                Você já possui a seguinte avaliação da base:

                {avaliacao}

                Você também já possui o seguinte perfil inicial da base:

                {profile_text}
                
                Use essas informações como ponto de partida.
                Use as ferramentas disponíveis apenas para transformar os dados
                na base.
                """),
            ],
            # Dataset a ser transformado
            'dataset':
            dataset,
            'tool_history':
            ToolHistory(),
            'frozen_columns':
            list()
        },
        config={"recursion_limit": qt_maxima_supersteps})
    return response["dataset"], response['tool_history']
