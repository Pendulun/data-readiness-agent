from dataclasses import dataclass
import pandas as pd
import streamlit as st
from timeit import default_timer as timer
from typing_extensions import Any, Dict, List, Tuple

from data_readyness_agent import agent, common_data_structs


@dataclass
class SidebarInputs:
    rodar_agente: bool
    dataframe: pd.DataFrame
    openai_api_key: str
    qt_max_iteracoes_agente_avaliacao: int
    target_col: str


@st.fragment
def download_dados_transformados(dados_transformados: pd.DataFrame):
    st.download_button(
        label="Download dados transformados em CSV",
        data=dados_transformados.to_csv().encode("utf-8"),
        file_name="transformed_data.csv",
        mime="text/csv",
        icon=":material/download:",
    )


def display_initial_info():
    st.header("Agente de Data Readyness")
    st.text(
        "Informe a base, escolha a coluna alvo e informe a sua OPENAI API KEY para avaliar a base!"
    )

    st.text(
        "Se estiver receoso de informar a sua chave da openai, sinta-se livre para verificar o código deste projeto (link abaixo)"
    )
    st.text(
        "O modelo usado internamente é o gpt-5-nano que é o mais barato de todos atualmente (Julho/2026)."
    )
    st.text(
        "O custo de avaliar a base deve ser de apenas alguns centavos de dólar caso a base seja grande."
    )

    st.markdown(
        "Criado por Daniel Souza de Campos. [Linkedin](https://www.linkedin.com/in/souzacamposdaniel/)  [GitHub](https://github.com/Pendulun)  [Repo](https://github.com/Pendulun/data-readyness-agent)"
    )


def display_sidebar() -> SidebarInputs:
    """
    Mostra os elementos do sidebar e retorna os seus inputs
    """
    target_col = None
    openai_api_key = None
    dataframe = None
    qt_max_iteracoes_agente = None
    rodar_agente = False

    with st.sidebar:
        uploaded_file = st.file_uploader("Base de dados",
                                         max_upload_size=10,
                                         type='csv',
                                         accept_multiple_files=False)

        all_cols = None
        if uploaded_file is not None:
            dataframe = pd.read_csv(uploaded_file)
            all_cols = dataframe.columns.tolist()

        target_col = st.selectbox("Coluna alvo",
                                  options=all_cols,
                                  index=None,
                                  disabled=all_cols is None)

        openai_api_key = st.text_input(label='OpenAI API key',
                                       type='password',
                                       persist_state='session',
                                       key='openai_api_key',
                                       disabled=uploaded_file is None)

        qt_max_iteracoes_agente = st.number_input(
            label=
            'Quantidade máxima de iterações do agente de avaliação da base',
            disabled=uploaded_file is None,
            value=0,
            min_value=0,
            max_value=50,
            step=1,
            help=
            "Isso não limita, necessariamente, a quantidade de tools chamadas")

        if len(openai_api_key) == 0:
            openai_api_key = None

        required_values = [uploaded_file, target_col, openai_api_key]
        can_generate_response = all(
            [val is not None for val in required_values])
        rodar_agente = st.button("Gerar resposta para variável alvo",
                                 disabled=not can_generate_response)

    return SidebarInputs(
        rodar_agente=rodar_agente,
        dataframe=dataframe,
        openai_api_key=openai_api_key,
        qt_max_iteracoes_agente_avaliacao=qt_max_iteracoes_agente,
        target_col=target_col)


def display_main_content(sidebar_inputs: SidebarInputs):
    """
    Mostra os elementos principais de acordo com as entradas do sidebar.
    Chama os agentes caso necessário
    """
    if sidebar_inputs.rodar_agente:
        display_agents_response(sidebar_inputs)

    if sidebar_inputs.dataframe is not None:
        st.subheader("Primeiras linhas do arquivo original:")
        st.write(sidebar_inputs.dataframe.head(10))


def display_agents_response(sidebar_inputs: SidebarInputs):
    """
    Chama os agentes de forma sequencial e mostra as suas respostas
    """
    st.subheader("Avaliação da base")
    start_time = timer()
    erro_ao_gerar_avaliacao = False
    try:
        avaliacao = call_eval_agent(sidebar_inputs)
    except Exception as e:
        erro_ao_gerar_avaliacao = True
        st.error("Um erro aconteceu ao gerar a avaliação!")
        st.error(e)

    if not erro_ao_gerar_avaliacao:
        st.success(f"Avaliação gerada em {timer() - start_time:.2f} segundos!")
        st.markdown(avaliacao.to_markdown())

        start_time = timer()
        erro_ao_gerar_transformacoes = False
        try:
            dados_transformados, tool_history = call_transformation_agent(
                sidebar_inputs, avaliacao)
        except Exception as e:
            erro_ao_gerar_transformacoes = True
            st.error("Um erro aconteceu ao tentar transformar a base!")
            st.error(e)

        if not erro_ao_gerar_transformacoes:
            st.success(
                f"Transformações aplicadas em {timer() - start_time:.2f} segundos!"
            )
            st.subheader("Primeiras linhas do arquivo transformado:")
            st.write(dados_transformados.head(10))
            download_dados_transformados(dados_transformados)
            st.subheader(
                "Histórico de uso de tools de transformação por coluna")
            st.write(tool_history)


def call_transformation_agent(
    sidebar_inputs: SidebarInputs,
    avaliacao: common_data_structs.EvalAgentResponse
) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, Any]]]]:
    """
    Consegue a resposta do agente de transformação da base
    """
    with st.spinner("[Passo 2/2] Transformando base...", show_time=True):
        dados_transformados, tool_history = agent.get_base_transformada(
            avaliacao.get_findings_str(),
            sidebar_inputs.dataframe,
            openai_api_key=sidebar_inputs.openai_api_key,
        )
    return dados_transformados, tool_history


def call_eval_agent(
        sidebar_inputs: SidebarInputs
) -> common_data_structs.EvalAgentResponse:
    """
    Consegue a resposta do agente de avaliação
    """
    with st.spinner("[Passo 1/2] Avaliando base...", show_time=True):
        avaliacao = agent.get_avaliacao(
            sidebar_inputs.dataframe,
            openai_api_key=sidebar_inputs.openai_api_key,
            qt_maxima_iteracoes_agente=sidebar_inputs.
            qt_max_iteracoes_agente_avaliacao,
            target_col=sidebar_inputs.target_col,
        )
    return avaliacao


if __name__ == "__main__":
    display_initial_info()
    sidebar_inputs = display_sidebar()
    display_main_content(sidebar_inputs)
