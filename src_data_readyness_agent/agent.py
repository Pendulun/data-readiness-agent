from dataclasses import dataclass
from langchain.messages import HumanMessage
from langsmith import Client, tracing_context
import pandas as pd
from typing_extensions import Tuple

from src_data_readyness_agent import config, security
from src_data_readyness_agent.common import data_structs
from src_data_readyness_agent.data_evaluation_agent import data_evaluation_agent
from src_data_readyness_agent.data_transformation_agent import data_transformation_agent
from src_data_readyness_agent.data_transformation_agent.data_structs import ToolHistory


@dataclass
class EvalAgentInputs():
    dataset: pd.DataFrame
    openai_api_key: str
    langsmith_api_key: str
    langsmith_project: str
    qt_maxima_iteracoes_agente: int
    target_col: str
    user_entry: str
    qt_maxima_supersteps: int
    prefered_language: str
    model: str

    def has_langsmith_configured(self):
        return self.langsmith_api_key is not None and len(
            self.langsmith_api_key
        ) > 0 and self.langsmith_project is not None and len(
            self.langsmith_project) > 0


@dataclass
class TransformAgentInputs():
    findings_str: str
    dataset: pd.DataFrame
    openai_api_key: str
    langsmith_api_key: str
    langsmith_project: str
    qt_max_iteracoes_agente: int
    qt_maxima_supersteps: int
    model: str

    def has_langsmith_configured(self):
        return self.langsmith_api_key is not None and len(
            self.langsmith_api_key
        ) > 0 and self.langsmith_project is not None and len(
            self.langsmith_project) > 0


def get_avaliacao(
        eval_inputs: EvalAgentInputs) -> data_structs.EvalAgentResponse:
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
        user_entry (str):
            Entrada manual do usuário contendo os seus objetivos
    """
    profile = create_dataset_profile(eval_inputs.dataset)
    avaliacao = generate_avaliacao(
        eval_inputs,
        profile,
    )
    return avaliacao


def get_base_transformada(
    transform_inputs: TransformAgentInputs,
) -> Tuple[pd.DataFrame, ToolHistory]:
    """
    Aplica transformações na base

    Args:
        transform_inputs (TransformAgentInputs):
            Os inputs necessários para chamar o agente de transformação

    Returns:
        Dataset transformado e o histórico de chamadas com sucesso a tools
        de transformação
    """
    # Conseguir o profile da base é barato então eu posso calcular aqui de novo
    profile = create_dataset_profile(transform_inputs.dataset)
    dataset_transformado, tool_history = get_transformed_df(
        transform_inputs, profile)

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
    eval_inputs: EvalAgentInputs,
    profile: data_structs.DatasetProfile,
) -> data_structs.EvalAgentResponse:
    """
    Invoca o agente responsável por gerar a avaliação da base
    """
    if eval_inputs.model not in config.ALLOWED_MODELS:
        if eval_inputs.prefered_language == 'English':
            raise ValueError(
                f"Model {eval_inputs.model} is not allowed for evaluation agent!"
            )
        elif eval_inputs.prefered_language == 'Português':
            raise ValueError(
                f"O modelo {eval_inputs.model} não é permitido no agente de avaliação!"
            )

    if eval_inputs.prefered_language == 'English':
        if security.PromptInjectionFilterEnglish().detect_injection(
                eval_inputs.user_entry):
            raise ValueError("Prompt injection attempt detected!")
    elif eval_inputs.prefered_language == 'Português':
        if security.PromptInjectionFilterPortuguese().detect_injection(
                eval_inputs.user_entry):
            raise ValueError("Tentativa de prompt injection detectada!")

    agent = data_evaluation_agent.get_agent(
        eval_inputs.openai_api_key, eval_inputs.qt_maxima_iteracoes_agente,
        eval_inputs.model)
    profile_text = profile.model_dump_json(indent=2)

    if len(eval_inputs.user_entry) == 0:
        eval_inputs.user_entry = "Nenhuma indicação do usuário"

    langsmith_client = None
    if eval_inputs.has_langsmith_configured():
        langsmith_client = Client(api_key=eval_inputs.langsmith_api_key)

    with tracing_context(name="Evaluation Agent",
                         enabled=eval_inputs.has_langsmith_configured(),
                         client=langsmith_client,
                         project_name=eval_inputs.langsmith_project,
                         tags=['evaluation_agent']):
        response = agent.invoke(
            {
                'messages': [
                    HumanMessage(content=f"""
                    Avalie a base de dados. Leve em consideração as informações recebidas.

                    PERFIL DA BASE:
                    {profile_text}
                    
                    COLUNA_ALVO_DO_MODELO:
                    {eval_inputs.target_col}

                    INDICAÇÃO DO USUÁRIO:
                    {eval_inputs.user_entry}
                    
                    DIRECIONAMENTOS:
                    1. Use essas informações como ponto de partida.
                    2. Não repita análises que já estão presentes no perfil.
                    3. Use as ferramentas disponíveis apenas para aprofundar
                    a investigação de possíveis problemas de qualidade.
                    4. Gere a resposta em {eval_inputs.prefered_language}
                    """),
                ],
                'dataset':
                eval_inputs.dataset,
                "dataset_profile":
                profile,
            },
            config={
                "recursion_limit": eval_inputs.qt_maxima_supersteps,
                "run_name": "Evaluation Agent"
            })
        final_response: data_structs.EvalAgentResponse = response[
            'structured_response']
    return final_response


def get_transformed_df(
        transform_inputs: TransformAgentInputs,
        profile: data_structs.DatasetProfile
) -> Tuple[pd.DataFrame, ToolHistory]:
    """
    Invoca o agente responsável por aplicar transformações sobre o dataset
    original de acordo com a avalição feita

    Returns:
        O dataset transformado e o histórico de uso de sucesso das tools de transformação
        da base por coluna
    """
    if transform_inputs.model not in config.ALLOWED_MODELS:
        raise ValueError(
            f"Model {transform_inputs.model} is not allowed for transformation agent!"
        )
    agent = data_transformation_agent.get_agent(
        transform_inputs.openai_api_key,
        transform_inputs.qt_max_iteracoes_agente, transform_inputs.model)
    profile_text = profile.model_dump_json(indent=2)

    langsmith_client = None
    if transform_inputs.has_langsmith_configured():
        langsmith_client = Client(api_key=transform_inputs.langsmith_api_key)

    response = None
    with tracing_context(name="Transformation agent",
                         enabled=transform_inputs.has_langsmith_configured(),
                         client=langsmith_client,
                         project_name=transform_inputs.langsmith_project,
                         tags=['transformation_agent']):
        response = agent.invoke(
            {
                'messages': [
                    HumanMessage(content=f"""
                Transforme essa base de dados.

                Você já possui a seguinte avaliação da base:

                {transform_inputs.findings_str}

                Você também já possui o seguinte perfil inicial da base:

                {profile_text}
                
                Use essas informações como ponto de partida.
                Use as ferramentas disponíveis apenas para transformar os dados
                na base.
                """),
                ],
                # Dataset a ser transformado
                'dataset':
                transform_inputs.dataset,
                'tool_history':
                ToolHistory(),
                'frozen_columns':
                set()
            },
            config={
                "recursion_limit": transform_inputs.qt_maxima_supersteps,
                "run_name": "Transformation Agent"
            })
    return response["dataset"], response['tool_history']
