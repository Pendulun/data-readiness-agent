from langchain.messages import HumanMessage
import pandas as pd
from typing_extensions import Any, Dict, List, Tuple

from data_readyness_agent import common_data_structs, data_evaluation_agent, data_transformation_agent


def create_dataset_profile(
        df: pd.DataFrame) -> common_data_structs.DatasetProfile:

    return common_data_structs.DatasetProfile(
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


def get_avaliacao(
    dataset: pd.DataFrame,
    profile: common_data_structs.DatasetProfile,
    openai_api_key: str,
    qt_maxima_iteracoes_agente: int,
    target_col: str,
) -> data_evaluation_agent.AgentResponse:

    agent = data_evaluation_agent.get_agent(
        openai_api_key,
        qt_maxima_iteracoes_agente,
    )
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


def get_transformed_df(
    avaliacao: str, profile: common_data_structs.DatasetProfile,
    dataset: pd.DataFrame, openai_api_key: str
) -> Tuple[str, pd.DataFrame, Dict[str, List[Dict[str, Any]]]]:
    """
    Aplica transformações sobre o dataset original de acordo com a avalição feita
    """
    agent = data_transformation_agent.get_agent(openai_api_key)
    profile_text = profile.model_dump_json(indent=2)

    response = agent.invoke({
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
        dict()
    })
    # final_response: data_evaluation_agent.AgentResponse = response[
    #     'structured_response'].to_markdown()
    final_response = ""
    return final_response, response["dataset"], response['tool_history']


def get_final_response(
    dataset: pd.DataFrame, openai_api_key: str,
    qt_maxima_iteracoes_agente: int, target_col: str
) -> Tuple[str, pd.DataFrame, Dict[str, List[Dict[str, Any]]]]:
    profile = create_dataset_profile(dataset)

    avaliacao = get_avaliacao(
        dataset,
        profile,
        openai_api_key=openai_api_key,
        qt_maxima_iteracoes_agente=qt_maxima_iteracoes_agente,
        target_col=target_col,
    )

    findings_str = avaliacao.get_findings_str()
    print(findings_str)
    transformacoes_aplicadas, dataset_transformado, tool_history = get_transformed_df(
        findings_str,
        profile,
        dataset,
        openai_api_key,
    )

    final_text = avaliacao.to_markdown() + "\n" + transformacoes_aplicadas

    return final_text, dataset_transformado, tool_history
