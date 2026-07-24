import pandas as pd
import streamlit as st
import time

from data_readyness_agent import agent

st.header("Agente de Data Readyness")
st.text(
    "Informe a base, escolha a coluna alvo e informe a sua OPEN API KEY para avaliar a base!"
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

required_values = None
target_col = None
openai_api_key = None
data_url = "user_data/dados_originais.csv"

with st.sidebar:
    uploaded_file = st.file_uploader("Base de dados",
                                     max_upload_size=10,
                                     type='csv',
                                     accept_multiple_files=False)

    all_cols = None
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        print("Salvando dados originais...")
        dataframe.to_csv(data_url, index=None)

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

    if len(openai_api_key) == 0:
        openai_api_key = None

    required_values = [uploaded_file, target_col, openai_api_key]
    can_generate_response = all([val is not None for val in required_values])
    gerar = st.button("Gerar resposta para variável alvo",
                      disabled=not can_generate_response)

if gerar:
    deu_erro = False
    st.subheader("Avaliação da base")
    with st.spinner("Gerando resposta...", show_time=True):
        try:
            resposta = agent.get_avaliacao(
                data_url,
                openai_api_key=openai_api_key,
            )
        except Exception as e:
            deu_erro = True
            resposta = str(e)
            raise e
    if not deu_erro:
        st.success("Resposta gerada!")
        st.markdown(resposta)
    else:
        st.error("Um erro aconteceu!")
        st.error(resposta)

if uploaded_file is not None:
    st.subheader("Primeiras linhas do arquivo:")
    st.write(dataframe.head(10))
