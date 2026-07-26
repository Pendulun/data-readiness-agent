import numpy as np
import pandas as pd
from langchain.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from typing_extensions import Any


@tool
def fill_col_na(column: str, fill_value: Any, runtime: ToolRuntime):
    """
    Imputa os valores nulos de uma coluna alvo com o valor informado e cria uma nova coluna booleana
    com o prefixo 'imputed_' indicando que o valor da coluna original naquela linha foi imputado"""
    df: pd.DataFrame = runtime.state["dataset"]
    return_msg = None
    try:
        is_na_mask = df[column].isna()
        df[column] = df[column].fillna(value=fill_value)
        df[f'imputed_{column}'] = np.where(is_na_mask, True, False)
        return_msg = f"Os valores nulos da coluna '{column}' foram imputados com {fill_value} com sucesso."
    except Exception as e:
        return_msg = f"Não foi possível imputar os valores nulos da coluna '{column}'! Erro: {str(e)}"
    finally:
        return Command(
            update={
                "dataset":
                df,
                "messages": [
                    ToolMessage(
                        content=(return_msg),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            })


@tool
def convert_column_to_datetime(
    column: str,
    runtime: ToolRuntime,
):
    """Converte uma coluna para o tipo datetime"""

    df: pd.DataFrame = runtime.state["dataset"]
    return_msg = None
    try:
        df[column] = pd.to_datetime(df[column], errors="coerce")
        return_msg = f"Os valores da coluna '{column}' foram transformados para datetime com sucesso."
    except Exception as e:
        return_msg = f"Não foi possível transformar os valores da coluna '{column}' para datetime! Erro: {str(e)}"
    finally:
        return Command(
            update={
                "dataset":
                df,
                "messages": [
                    ToolMessage(
                        content=(return_msg),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            })


@tool
def convert_column_to_int(
    column: str,
    runtime: ToolRuntime,
):
    """Converte uma coluna para o tipo inteiro"""

    df: pd.DataFrame = runtime.state["dataset"]
    return_msg = None
    try:
        df[column] = df[column].astype("int")
        return_msg = f"Os valores da coluna '{column}' foram convertidos para inteiro com sucesso."
    except Exception as e:
        return_msg = f"Não foi possível converter os valores da coluna '{column}' para inteiro! Erro: {str(e)}"
    finally:
        return Command(
            update={
                "dataset":
                df,
                "messages": [
                    ToolMessage(
                        content=(return_msg),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            })


@tool
def convert_column_to_float(
    column: str,
    runtime: ToolRuntime,
):
    """Converte uma coluna para o tipo float"""

    df: pd.DataFrame = runtime.state["dataset"]
    return_msg = None
    try:
        df[column] = df[column].astype("float")
        return_msg = f"Os valores da coluna '{column}' foram convertidos para float com sucesso."
    except Exception as e:
        return_msg = f"Não foi possível converter os valores da coluna '{column}' para float! Erro: {str(e)}"
    finally:
        return Command(
            update={
                "dataset":
                df,
                "messages": [
                    ToolMessage(
                        content=(return_msg),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            })
