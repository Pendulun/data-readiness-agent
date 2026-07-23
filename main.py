import pandas as pd
import streamlit as st
from data_readyness_agent import agent

uploaded_file = st.file_uploader("Base de dados", max_upload_size=10, type='csv', accept_multiple_files=False)
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    print("Salvando dados originais...")
    data_url = "user_data/dados_originais.csv"
    dataframe.to_csv(data_url, index=None)
    st.write(dataframe)

    st.write(agent.get_avaliacao(data_url))