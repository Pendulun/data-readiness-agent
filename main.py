from dataclasses import dataclass
import json
import pandas as pd
from pathlib import Path
import streamlit as st
from timeit import default_timer as timer
from typing_extensions import Tuple

from src_data_readyness_agent import agent, config
from src_data_readyness_agent.common.data_structs import EvalAgentResponse
from src_data_readyness_agent.data_transformation_agent.data_structs import ToolHistory


@dataclass
class SidebarInputs:
    rodar_agente: bool
    dataframe: pd.DataFrame
    openai_api_key: str
    langsmith_api_key: str
    langsmith_project: str
    eval_agent_model: str
    transform_agent_model: str
    qt_max_iteracoes_agente_avaliacao: int
    qt_maxima_supersteps_avaliacao: int
    qt_max_iteracoes_agente_transformacao: int
    qt_maxima_supersteps_transformacao: int
    target_col: str
    user_entry: str


@st.fragment
def download_dados_transformados(dados_transformados: pd.DataFrame,
                                 translation: dict):
    st.download_button(
        label=translation["download_button_label"],
        data=dados_transformados.to_csv(index=None).encode("utf-8"),
        file_name="transformed_data.csv",
        mime="text/csv",
        icon=":material/download:",
    )


def display_initial_info(translations: dict):
    st.header(translations['main_header'])
    st.text(translations["info1"])

    st.text(translations["info2"])
    st.text(translations["info3"])
    st.text(translations["info4"])

    st.markdown(
        translations["contact"] +
        "Daniel Souza de Campos. [Linkedin](https://www.linkedin.com/in/souzacamposdaniel/)  [GitHub](https://github.com/Pendulun)  [Repo](https://github.com/Pendulun/data-readyness-agent)"
    )


def display_sidebar(translation: dict) -> SidebarInputs:
    """
    Mostra os elementos do sidebar e retorna os seus inputs
    """
    target_col = None
    openai_api_key = None
    dataframe = None
    qt_max_iteracoes_agente_avaliacao = None
    rodar_agente = False

    with st.sidebar:
        uploaded_file = st.file_uploader(translation["file_upload_label"],
                                         max_upload_size=10,
                                         type='csv',
                                         accept_multiple_files=False,
                                         help=translation["file_upload_help"])

        all_cols = None
        if uploaded_file is not None:
            ext = Path(uploaded_file.name).suffix.lower()
            if ext == ".csv":
                sep = st.text_input(label=translation["csv_file_sep_label"],
                                    value=",")
                dataframe = pd.read_csv(uploaded_file, delimiter=sep)

            all_cols = dataframe.columns.tolist()

        target_col = st.selectbox(
            translation["target_col_label"],
            options=all_cols,
            index=None,
            disabled=all_cols is None,
            placeholder=translation['target_col_placeholder'])

        user_entry = st.text_area(translation['user_entry_label'],
                                  value="",
                                  max_chars=1000,
                                  key='user_entry',
                                  disabled=uploaded_file is None,
                                  persist_state='session')

        openai_api_key = st.text_input(
            label=translation["openai_api_key_label"],
            type='password',
            persist_state='session',
            key='openai_api_key',
            disabled=uploaded_file is None)

        qt_max_iteracoes_agente_avaliacao = st.number_input(
            label=translation["eval_agent_max_its_label"],
            disabled=uploaded_file is None,
            value=15,
            min_value=1,
            max_value=1000,
            step=1,
            persist_state='session',
            key="eval_agent_max_its",
            help=translation["eval_agent_max_its_help"])

        qt_max_iteracoes_agente_transformacao = st.number_input(
            label=translation["transform_agent_max_its_label"],
            disabled=uploaded_file is None,
            value=50,
            min_value=1,
            max_value=1000,
            step=1,
            persist_state='session',
            key="transform_agent_max_its",
            help=translation["transform_agent_max_its_help"])

        if len(openai_api_key) == 0:
            openai_api_key = None

        required_values = [uploaded_file, target_col, openai_api_key]
        can_generate_response = all(
            [val is not None for val in required_values])

        with st.expander(translation["advanced_options_label"]):
            st.write(translation["advanced_options_info"])
            qt_maxima_supersteps_avaliacao = st.number_input(
                label=translation["eval_agent_max_supersteps_label"],
                disabled=uploaded_file is None,
                min_value=int(qt_max_iteracoes_agente_avaliacao * 3),
                max_value=max(int(qt_max_iteracoes_agente_avaliacao * 5), 100),
                value='min',
                step=1,
                persist_state='session',
                key="eval_agent_max_supersteps",
                help=translation["eval_agent_max_supersteps_help"])
            qt_maxima_supersteps_transformacao = st.number_input(
                label=translation["transform_agent_max_supersteps_label"],
                disabled=uploaded_file is None,
                min_value=int(qt_max_iteracoes_agente_transformacao * 3),
                max_value=max(int(qt_max_iteracoes_agente_transformacao * 5),
                              100),
                value='min',
                step=1,
                persist_state='session',
                key="transform_agent_max_supersteps",
                help=translation["transform_agent_max_supersteps_help"])

            st.markdown(translation['advcd_opts_models_info'])
            eval_model = st.selectbox(
                label=translation['eval_model_selectbox_label'],
                options=config.ALLOWED_MODELS,
                index=0,
                persist_state='session',
                key="base_eval_model",
                disabled=uploaded_file is None,
            )
            transform_model = st.selectbox(
                label=translation['transform_model_selectbox_label'],
                options=config.ALLOWED_MODELS,
                index=0,
                persist_state='session',
                key="base_transform_model",
                disabled=uploaded_file is None,
            )
            st.write(translation['langsmith_info_label'])
            langsmith_api_key = st.text_input(
                label=translation["langsmith_api_key_label"],
                value="",
                type='password',
                persist_state='session',
                key='langsmith_api_key',
                disabled=uploaded_file is None)
            langsmith_project = st.text_input(
                label=translation["langsmith_project_label"],
                value="",
                persist_state='session',
                key='langsmith_project',
                disabled=uploaded_file is None)

        rodar_agente = st.button(translation["generate_response_label"],
                                 disabled=not can_generate_response)
    return SidebarInputs(
        rodar_agente=rodar_agente,
        dataframe=dataframe,
        eval_agent_model=eval_model,
        transform_agent_model=transform_model,
        openai_api_key=openai_api_key,
        langsmith_api_key=langsmith_api_key,
        langsmith_project=langsmith_project,
        qt_max_iteracoes_agente_avaliacao=qt_max_iteracoes_agente_avaliacao,
        qt_max_iteracoes_agente_transformacao=
        qt_max_iteracoes_agente_transformacao,
        qt_maxima_supersteps_avaliacao=qt_maxima_supersteps_avaliacao,
        qt_maxima_supersteps_transformacao=qt_maxima_supersteps_transformacao,
        target_col=target_col,
        user_entry=user_entry)


def display_main_content(sidebar_inputs: SidebarInputs, translation: dict,
                         prefered_language: str):
    """
    Mostra os elementos principais de acordo com as entradas do sidebar.
    Chama os agentes caso necessário
    """
    if sidebar_inputs.rodar_agente:
        display_agents_response(sidebar_inputs, translation, prefered_language)

    if sidebar_inputs.dataframe is not None:
        st.subheader(translation["original_df_subheader"])
        st.write(sidebar_inputs.dataframe.head(10))


def display_agents_response(sidebar_inputs: SidebarInputs, translation: dict,
                            prefered_language: str):
    """
    Chama os agentes de forma sequencial e mostra as suas respostas
    """
    st.subheader(translation["evaluation_subheader"])
    start_time = timer()
    erro_ao_gerar_avaliacao = False
    try:
        avaliacao = call_eval_agent(sidebar_inputs, translation,
                                    prefered_language)
    except Exception as e:
        erro_ao_gerar_avaliacao = True
        st.error(translation["evaluation_error_msg"])
        st.error(e)

    if not erro_ao_gerar_avaliacao:
        st.success(translation["evaluation_success_msg_fmt"].format(
            seconds=round(timer() - start_time, 2)))
        st.markdown(avaliacao.to_markdown())

        start_time = timer()
        erro_ao_gerar_transformacoes = False
        try:
            dados_transformados, tool_history = call_transformation_agent(
                sidebar_inputs, avaliacao, translation)
        except Exception as e:
            erro_ao_gerar_transformacoes = True
            st.error(translation["transformation_error_msg"])
            st.error(e)

        if not erro_ao_gerar_transformacoes:
            st.success(translation["transformation_success_msg_fmt"].format(
                seconds=round(timer() - start_time, 2)))
            st.subheader(translation["transformed_data_subheader"])
            st.write(dados_transformados.head(10))
            download_dados_transformados(dados_transformados, translation)
            st.subheader(translation["tools_history_subheader"])
            st.write(tool_history.history_as_dict())
            st.subheader(translation["tools_per_col_history_subheader"])
            st.write(tool_history.col_transformation_history_as_dict())
            st.subheader(translation["tools_usage_info_subheader"])
            st.write(translation['basic_tool_stats_text_fmt'].format(
                n_tools_called=tool_history.n_tools_called(),
                n_all_calls=tool_history.n_all_tool_calls()))
            tool_stats_df = pd.DataFrame({
                translation['n_calls_per_tool']:
                tool_history.n_calls_per_tool(),
                translation['tools_sucess_rate']: {
                    tool: round(suc_rate * 100, 2)
                    for tool, suc_rate in
                    tool_history.sucess_rate_per_tool().items()
                }
            })
            st.dataframe(tool_stats_df)


def call_transformation_agent(
        sidebar_inputs: SidebarInputs, avaliacao: EvalAgentResponse,
        translation: dict) -> Tuple[pd.DataFrame, ToolHistory]:
    """
    Consegue a resposta do agente de transformação da base
    """
    with st.spinner(translation["transformation_spinner_label"],
                    show_time=True):
        transform_inputs = agent.TransformAgentInputs(
            findings_str=avaliacao.get_findings_str(),
            dataset=sidebar_inputs.dataframe,
            openai_api_key=sidebar_inputs.openai_api_key,
            langsmith_api_key=sidebar_inputs.langsmith_api_key,
            langsmith_project=sidebar_inputs.langsmith_project,
            qt_max_iteracoes_agente=sidebar_inputs.
            qt_max_iteracoes_agente_transformacao,
            qt_maxima_supersteps=sidebar_inputs.
            qt_maxima_supersteps_transformacao,
            model=sidebar_inputs.transform_agent_model)
        dados_transformados, tool_history = agent.get_base_transformada(
            transform_inputs)
    return dados_transformados, tool_history


def call_eval_agent(sidebar_inputs: SidebarInputs, translation: str,
                    prefered_language: str) -> EvalAgentResponse:
    """
    Consegue a resposta do agente de avaliação
    """
    with st.spinner(translation["evaluation_spinner_label"], show_time=True):
        eval_inputs = agent.EvalAgentInputs(
            dataset=sidebar_inputs.dataframe,
            openai_api_key=sidebar_inputs.openai_api_key,
            langsmith_api_key=sidebar_inputs.langsmith_api_key,
            langsmith_project=sidebar_inputs.langsmith_project,
            qt_maxima_iteracoes_agente=sidebar_inputs.
            qt_max_iteracoes_agente_avaliacao,
            target_col=sidebar_inputs.target_col,
            user_entry=sidebar_inputs.user_entry,
            qt_maxima_supersteps=sidebar_inputs.qt_maxima_supersteps_avaliacao,
            prefered_language=prefered_language,
            model=sidebar_inputs.eval_agent_model)
        avaliacao = agent.get_avaliacao(eval_inputs)
    return avaliacao


if __name__ == "__main__":
    language = st.sidebar.selectbox("Idioma / Language",
                                    options=["Português", "English"])

    if language == 'Português':
        with open("translations/pt-BR.json", "r", encoding="utf-8") as f:
            translations = json.load(f)
    else:
        with open("translations/en.json", "r", encoding="utf-8") as f:
            translations = json.load(f)

    display_initial_info(translations['initial_info'])
    sidebar_inputs = display_sidebar(translations['sidebar'])
    display_main_content(sidebar_inputs, translations['main_content'],
                         language)
