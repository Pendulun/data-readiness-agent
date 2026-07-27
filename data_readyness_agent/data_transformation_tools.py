import copy
import numpy as np
import pandas as pd
from langchain.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from typing_extensions import Any, List


@tool
def get_n_cols(runtime: ToolRuntime) -> int:
    """Retorna a quantidade de colunas na base"""
    return runtime.state['dataset'].shape[1]


@tool
def get_cols_names(runtime: ToolRuntime) -> List[str]:
    """Retorna os nomes das colunas na base"""
    return runtime.state['dataset'].columns.tolist()


@tool
def has_nulls(column: str, runtime: ToolRuntime) -> int:
    """Retorna True ou False se uma coluna alvo na base possui valores nulos"""
    if column not in runtime.state['dataset'].columns.tolist():
        return {"error": f"A coluna '{column}' não existe na base."}
    return runtime.state['dataset'][column].isna().sum() > 0


@tool
def get_unique_counts(column: str, runtime: ToolRuntime) -> int:
    """Retorna a quantidade de valores únicos em uma coluna alvo na base"""
    if column not in runtime.state['dataset'].columns.tolist():
        return {"error": f"A coluna '{column}' não existe na base."}
    return runtime.state['dataset'][column].nunique()


@tool
def get_column_type(column: str, runtime: ToolRuntime) -> int:
    """Retorna o tipo dos valores de uma coluna alvo na base"""
    if column not in runtime.state['dataset'].columns.tolist():
        return {"error": f"A coluna '{column}' não existe na base."}
    return str(runtime.state['dataset'][column].dtype)


@tool
def get_col_unique_preview(column: str, runtime: ToolRuntime) -> List[Any]:
    """Retorna um preview dos valores únicos contidos em uma coluna alvo da base"""
    if column not in runtime.state['dataset'].columns.tolist():
        return {"error": f"Coluna {column} não está presente na base"}
    serie: pd.Series = runtime.state['dataset'][column]
    return serie.drop_duplicates().head().tolist()


@tool
def replace_substring(column: str, old_substring: str, new_substring: str,
                      runtime: ToolRuntime):
    """
    Aplica o replace de uma substring em valores contidos em uma coluna alvo por um novo valor.
    Por exemplo: column:'faturamento', old_substring:'R$', new_substring:'' vai substituir 'R$' por
    nada na coluna 'faturamento'
    """
    df: pd.DataFrame = runtime.state["dataset"]
    if column not in runtime.state['dataset'].columns.tolist():
        return {"error": f"Coluna {column} não está presente na base"}
    new_tool_history: dict = copy.deepcopy(runtime.state['tool_history'])
    try:
        df[column] = df[column].astype(str).str.replace(old_substring,
                                                        new_substring,
                                                        regex=False)
        return_msg = f"A substring '{old_substring}' da coluna '{column}' foi substituida por '{new_substring}' com sucesso."
    except Exception as e:
        return_msg = f"Não foi possível substituir a substring  '{old_substring}' na coluna '{column}'! Erro: {str(e)}"
    else:
        new_tool_history.setdefault(column, list()).append({
            'tool': 'replace_substring',
            'args': {
                'old_substring': old_substring,
                'new_substring': new_substring
            }
        })

    finally:
        return Command(
            update={
                "dataset":
                df,
                'tool_history':
                new_tool_history,
                "messages": [
                    ToolMessage(
                        content=(return_msg),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            })


@tool
def drop_column(column: str, runtime: ToolRuntime):
    """Remove uma coluna alvo da base"""
    df: pd.DataFrame = runtime.state["dataset"]
    if column not in runtime.state['dataset'].columns.tolist():
        return {"error": f"Coluna {column} não está presente na base"}
    df = df.drop(columns=[column])
    return_msg = f"A coluna '{column}' foi removida com sucesso."
    new_tool_history: dict = copy.deepcopy(runtime.state['tool_history'])
    new_tool_history.setdefault(column, list()).append({
        'tool': 'drop_column',
    })

    return Command(
        update={
            "dataset":
            df,
            'tool_history':
            new_tool_history,
            "messages": [
                ToolMessage(
                    content=(return_msg),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        })


@tool
def fill_col_na(column: str, fill_value: Any, runtime: ToolRuntime):
    """
    Imputa os valores nulos de uma coluna alvo com o valor informado e cria uma nova coluna booleana
    com o prefixo 'imputed_' indicando que o valor da coluna original naquela linha foi imputado"""
    df: pd.DataFrame = runtime.state["dataset"]
    if column not in runtime.state['dataset'].columns.tolist():
        return {"error": f"Coluna {column} não está presente na base"}
    return_msg = None
    new_tool_history: dict = copy.deepcopy(runtime.state['tool_history'])
    try:
        is_na_mask = df[column].isna()
        df[column] = df[column].fillna(value=fill_value)
        df[f'imputed_{column}'] = np.where(is_na_mask, True, False)
        return_msg = f"Os valores nulos da coluna '{column}' foram imputados com {fill_value} com sucesso."
    except Exception as e:
        return_msg = f"Não foi possível imputar os valores nulos da coluna '{column}'! Erro: {str(e)}"
    else:
        new_tool_history.setdefault(column, list()).append({
            'tool': 'fill_col_na',
            'args': {
                'fill_value': fill_value,
            }
        })
    finally:
        return Command(
            update={
                "dataset":
                df,
                'tool_history':
                new_tool_history,
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
    if column not in runtime.state['dataset'].columns.tolist():
        return {"error": f"Coluna {column} não está presente na base"}
    return get_command_with_converted_column_type(df, column, 'datetime',
                                                  runtime)


@tool
def convert_column_to_int(
    column: str,
    runtime: ToolRuntime,
):
    """Converte uma coluna para o tipo inteiro"""
    df: pd.DataFrame = runtime.state["dataset"]
    if column not in runtime.state['dataset'].columns.tolist():
        return {"error": f"Coluna {column} não está presente na base"}
    return get_command_with_converted_column_type(df, column, 'int', runtime)


@tool
def convert_column_to_float(
    column: str,
    runtime: ToolRuntime,
):
    """Converte uma coluna para o tipo float"""
    df: pd.DataFrame = runtime.state["dataset"]
    if column not in runtime.state['dataset'].columns.tolist():
        return {"error": f"Coluna {column} não está presente na base"}
    return get_command_with_converted_column_type(df, column, 'float', runtime)


def get_command_with_converted_column_type(
    df: pd.DataFrame,
    column: str,
    target_type: str,
    runtime: ToolRuntime,
) -> Command:
    """
    Retorna um comando para atualizar o dataset com a coluna transformada. Se falhar, a conversão 
    não é realizada
    """
    return_msg = None

    transform_func = lambda x, _type: x.astype(_type)
    if target_type == 'datetime':
        transform_func = lambda x, _type: pd.to_datetime(x, errors="coerce")
    new_tool_history: dict = copy.deepcopy(runtime.state['tool_history'])

    try:
        df[column] = transform_func(df[column], target_type)
        return_msg = f"Os valores da coluna '{column}' foram convertidos para {target_type} com sucesso."
    except Exception as e:
        return_msg = f"Não foi possível converter os valores da coluna '{column}' para {target_type}! Erro: {str(e)}"
    else:
        new_tool_history.setdefault(column, list()).append(
            {'tool': f'convert_column_to_{target_type}'})
    finally:
        return Command(
            update={
                "dataset":
                df,
                'tool_history':
                new_tool_history,
                "messages": [
                    ToolMessage(
                        content=(return_msg),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            })
