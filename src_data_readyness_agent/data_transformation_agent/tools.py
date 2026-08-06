import numpy as np
import pandas as pd
from langchain.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from typing_extensions import Any, Dict, List, Union

from src_data_readyness_agent.data_transformation_agent.utils import track_history, ToolType


# Utils
def raise_if_column_not_in_dataset(dataset: pd.DataFrame, column: str):
    if column not in dataset.columns.tolist():
        raise ValueError(f"A coluna '{column}' não existe na base.")


def raise_if_column_in_dataset(dataset: pd.DataFrame, column: str):
    if column in dataset.columns.tolist():
        raise ValueError(f"A coluna '{column}' já existe na base.")


def raise_if_column_should_be_frozen(column: str, frozen_columns: set[str]):
    if column in frozen_columns:
        raise ValueError(f"A coluna {column} nunca pode ser modificada!")


# Tools
@tool()
@track_history(tool_type=ToolType.QUERY)
def get_n_cols(runtime: ToolRuntime) -> int:
    """Retorna a quantidade de colunas na base"""
    return runtime.state['dataset'].shape[1]


@tool
@track_history(tool_type=ToolType.QUERY)
def get_cols_names(runtime: ToolRuntime) -> List[str]:
    """Retorna os nomes das colunas na base"""
    return runtime.state['dataset'].columns.tolist()


@tool()
@track_history(tool_type=ToolType.QUERY)
def has_nulls(column: str, runtime: ToolRuntime) -> int:
    """Retorna True ou False se uma coluna alvo na base possui valores nulos"""
    raise_if_column_not_in_dataset(runtime.state['dataset'], column)
    return runtime.state['dataset'][column].isna().sum() > 0


@tool
@track_history(tool_type=ToolType.QUERY)
def get_unique_counts(column: str, runtime: ToolRuntime) -> int:
    """Retorna a quantidade de valores únicos em uma coluna alvo na base"""
    raise_if_column_not_in_dataset(runtime.state['dataset'], column)
    return runtime.state['dataset'][column].nunique()


@tool
@track_history(tool_type=ToolType.QUERY)
def get_column_type(column: str, runtime: ToolRuntime) -> int:
    """Retorna o tipo dos valores de uma coluna alvo na base"""
    raise_if_column_not_in_dataset(runtime.state['dataset'], column)
    return str(runtime.state['dataset'][column].dtype)


@tool
@track_history(tool_type=ToolType.QUERY)
def get_col_unique(column: str, runtime: ToolRuntime) -> List[Any]:
    """Retorna todos os valores únicos contidos em uma coluna alvo da base"""
    raise_if_column_not_in_dataset(runtime.state['dataset'], column)
    serie: pd.Series = runtime.state['dataset'][column]
    return serie.unique().tolist()


@tool
@track_history(tool_type=ToolType.CREATE_COL, modified_col='column')
def rename_column(column: str, new_column: str, runtime: ToolRuntime):
    """
    Renomeia uma coluna na base.
    """
    df: pd.DataFrame = runtime.state["dataset"]

    raise_if_column_not_in_dataset(df, column)
    raise_if_column_should_be_frozen(column, runtime.state['frozen_columns'])
    raise_if_column_in_dataset(df, new_column)

    df.rename(columns={column: new_column}, inplace=True)

    return_msg = f"Coluna {column} renomeada com sucesso para {new_column}"

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
@track_history(tool_type=ToolType.TRANSFORM_COL, modified_col='column')
def replace_substring(column: str, old_substring: str, new_substring: str,
                      runtime: ToolRuntime):
    """
    Aplica o replace de uma substring em valores contidos em uma coluna alvo por um novo valor.
    Por exemplo: column:'faturamento', old_substring:'R$', new_substring:'' vai substituir 'R$' por
    nada na coluna 'faturamento'
    """
    df: pd.DataFrame = runtime.state["dataset"]
    raise_if_column_not_in_dataset(df, column)
    raise_if_column_should_be_frozen(column, runtime.state['frozen_columns'])
    if old_substring == "":
        raise ValueError(
            "old_substring não pode ser uma string vazia! Se precisar substituir uma string vazia, use a tool de mapeamento de valores!"
        )

    try:
        df[column] = df[column].astype(str).str.replace(old_substring,
                                                        new_substring,
                                                        regex=False)
        return_msg = f"A substring '{old_substring}' da coluna '{column}' foi substituida por '{new_substring}' com sucesso."
    except Exception as e:
        raise ValueError(
            f"Não foi possível substituir a substring  '{old_substring}' na coluna '{column}'! Erro: {str(e)}"
        )
    else:
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
@track_history(tool_type=ToolType.DROP_COL, modified_col='column')
def drop_column(column: str, runtime: ToolRuntime):
    """Remove uma coluna alvo da base"""
    df: pd.DataFrame = runtime.state["dataset"]
    raise_if_column_not_in_dataset(df, column)
    raise_if_column_should_be_frozen(column, runtime.state['frozen_columns'])
    df = df.drop(columns=[column])
    return_msg = f"A coluna '{column}' foi removida com sucesso."

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
@track_history(tool_type=ToolType.TRANSFORM_COL, modified_col='column')
def fill_col_na(column: str, fill_value: Any, runtime: ToolRuntime):
    """
    Imputa os valores nulos de uma coluna alvo com o valor informado e cria uma nova coluna booleana
    com o prefixo 'imputed_' indicando que o valor da coluna original naquela linha foi imputado"""
    df: pd.DataFrame = runtime.state["dataset"]
    raise_if_column_not_in_dataset(df, column)
    raise_if_column_should_be_frozen(column, runtime.state['frozen_columns'])
    return_msg = None
    try:
        is_na_mask = df[column].isna()
        df[column] = df[column].fillna(value=fill_value)
        new_col_name = f'imputed_{column}'
        df[new_col_name] = np.where(is_na_mask, True, False)
        frozen_columns: set = runtime.state['frozen_columns'].copy()
        frozen_columns.add(new_col_name)
        return_msg = f"Os valores nulos da coluna '{column}' foram imputados com {fill_value} com sucesso."
    except Exception as e:
        raise ValueError(
            f"Não foi possível imputar os valores nulos da coluna '{column}'! Erro: {str(e)}"
        )
    else:
        return Command(
            update={
                "dataset":
                df,
                "frozen_columns":
                frozen_columns,
                "messages": [
                    ToolMessage(
                        content=(return_msg),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            })


@tool
@track_history(tool_type=ToolType.TRANSFORM_COL, modified_col='column')
def convert_column_to_datetime(
    column: str,
    runtime: ToolRuntime,
):
    """Converte uma coluna para o tipo datetime"""
    return get_command_with_converted_column_type(column, 'datetime', runtime)


@tool
@track_history(tool_type=ToolType.TRANSFORM_COL, modified_col='column')
def convert_column_to_int(
    column: str,
    runtime: ToolRuntime,
):
    """Converte uma coluna para o tipo inteiro"""
    return get_command_with_converted_column_type(column, 'int', runtime)


@tool
@track_history(tool_type=ToolType.TRANSFORM_COL, modified_col='column')
def convert_column_to_float(
    column: str,
    runtime: ToolRuntime,
):
    """Converte uma coluna para o tipo float"""
    return get_command_with_converted_column_type(column, 'float', runtime)


def get_command_with_converted_column_type(
    column: str,
    target_type: str,
    runtime: ToolRuntime,
) -> Command:
    """
    Retorna um comando para atualizar o dataset com a coluna transformada. Se falhar, a conversão 
    não é realizada
    """
    df: pd.DataFrame = runtime.state["dataset"]
    raise_if_column_not_in_dataset(df, column)
    raise_if_column_should_be_frozen(column, runtime.state['frozen_columns'])

    return_msg = None

    transform_func = lambda x, _type: x.astype(_type)
    if target_type == 'datetime':
        transform_func = lambda x, _type: pd.to_datetime(x, errors="coerce")

    try:
        df[column] = transform_func(df[column], target_type)
        return_msg = f"Os valores da coluna '{column}' foram convertidos para {target_type} com sucesso."
    except Exception as e:
        raise ValueError(
            f"Não foi possível converter os valores da coluna '{column}' para {target_type}! Erro: {str(e)}"
        )
    else:
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
@track_history(tool_type=ToolType.CREATE_COL, modified_col='new_col_name')
def subtract_cols(
    first_col: str,
    second_col: str,
    new_col_name: str,
    runtime: ToolRuntime,
):
    """
    Subtrai duas colunas da base criando uma nova.
    Exemplo:
    - first_col = 'faturamento_antes'
    - second_col = 'faturamento_depois'
    - new_col_name = 'diff_faturamento'

    Isso faz df[new_col_name] = df[first_col] - df[second_col]
    """
    return create_new_col_using_two_others(first_col, second_col, new_col_name,
                                           'subtraction', runtime)


@tool
@track_history(tool_type=ToolType.CREATE_COL, modified_col='new_col_name')
def sum_cols(
    first_col: str,
    second_col: str,
    new_col_name: str,
    runtime: ToolRuntime,
):
    """
    Soma duas colunas da base criando uma nova.
    Exemplo:
    - first_col = 'faturamento_mes_1'
    - second_col = 'faturamento_mes_2'
    - new_col_name = 'faturamento_final'

    Isso faz df[new_col_name] = df[first_col] + df[second_col]
    """
    return create_new_col_using_two_others(first_col, second_col, new_col_name,
                                           'sum', runtime)


@tool
@track_history(tool_type=ToolType.CREATE_COL, modified_col='new_col_name')
def divide_cols(
    first_col: str,
    second_col: str,
    new_col_name: str,
    runtime: ToolRuntime,
):
    """
    Divide duas colunas da base criando uma nova de forma segura
    Exemplo:
    - first_col = 'faturamento'
    - second_col = 'metragem'
    - new_col_name = 'faturamento_por_metro'

    Isso faz df[new_col_name] = df[first_col] / df[second_col].fillna(1)
    """
    return create_new_col_using_two_others(first_col, second_col, new_col_name,
                                           'div', runtime)


@tool
@track_history(tool_type=ToolType.CREATE_COL, modified_col='new_col_name')
def multiply_cols(
    first_col: str,
    second_col: str,
    new_col_name: str,
    runtime: ToolRuntime,
):
    """
    Multiplica duas colunas da base criando uma nova de forma segura
    Exemplo:
    - first_col = 'comprimento'
    - second_col = 'altura'
    - new_col_name = 'area'

    Isso faz df[new_col_name] = df[first_col] * df[second_col]
    """
    return create_new_col_using_two_others(first_col, second_col, new_col_name,
                                           'prod', runtime)


@tool
@track_history(tool_type=ToolType.CREATE_COL, modified_col='new_col_name')
def power_cols(
    first_col: str,
    second_col: str,
    new_col_name: str,
    runtime: ToolRuntime,
):
    """
    Eleva os valores da primeira coluna pelos valores da segunda coluna criando uma nova
    Isso faz df[new_col_name] = df[first_col] ** df[second_col]
    """
    return create_new_col_using_two_others(first_col, second_col, new_col_name,
                                           'prod', runtime)


def create_new_col_using_two_others(
    first_col: str,
    second_col: str,
    new_col_name: str,
    op_type: str,
    runtime: ToolRuntime,
) -> Command:
    """
    Retorna um comando para atualizar o dataset com a coluna transformada. Se falhar, a conversão 
    não é realizada
    """
    df: pd.DataFrame = runtime.state["dataset"]
    for col in [first_col, second_col]:
        raise_if_column_not_in_dataset(df, col)

    raise_if_column_in_dataset(df, new_col_name)

    return_msg = None

    op_to_func = {
        'subtraction': (lambda x: x[first_col] - x[second_col]),
        'sum': (lambda x: x[first_col] + x[second_col]),
        'div': (lambda x: x[first_col] / np.maximum(x[second_col], 1)),
        'prod': (lambda x: x[first_col] * x[second_col]),
        'pow': (lambda x: x[first_col]**x[second_col])
    }

    try:
        df[new_col_name] = op_to_func[op_type](df)
        return_msg = f"Os valores das colunas '{first_col}' e '{second_col}' foram usados para criar {new_col_name} com sucesso."
    except Exception as e:
        raise ValueError(
            f"Não foi possível realizar a operação {op_type} usando valores das coluna '{first_col}' e {second_col}! Erro: {str(e)}"
        )
    else:
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
@track_history(tool_type=ToolType.TRANSFORM_COL, modified_col='column')
def subtract_value(
    column: str,
    value: Union[int, float],
    runtime: ToolRuntime,
):
    """
    Subtrai um valor fixo de uma coluna da base
    Exemplo:
    - column = 'idade'
    - value = 2

    Isso faz df[column] = df[column] - value
    """
    return apply_constant_to_col(column, value, 'subtraction', runtime)


@tool
@track_history(tool_type=ToolType.TRANSFORM_COL, modified_col='column')
def sum_value(
    column: str,
    value: Union[int, float],
    runtime: ToolRuntime,
):
    """
    Soma um valor fixo de uma coluna da base
    Exemplo:
    - column = 'idade'
    - value = 2

    Isso faz df[column] = df[column] + value
    """
    return apply_constant_to_col(column, value, 'sum', runtime)


@tool
@track_history(tool_type=ToolType.TRANSFORM_COL, modified_col='column')
def divide_value(
    column: str,
    value: Union[int, float],
    runtime: ToolRuntime,
):
    """
    Divide uma coluna da base por um valor fixo
    Exemplo:
    - column = 'idade'
    - value = 2

    Isso faz df[column] = df[column] / value
    """
    return apply_constant_to_col(column, value, 'div', runtime)


@tool
@track_history(tool_type=ToolType.TRANSFORM_COL, modified_col='column')
def multiply_value(
    column: str,
    value: Union[int, float],
    runtime: ToolRuntime,
):
    """
    Multiplica uma coluna da base por um valor fixo
    Exemplo:
    - column = 'idade'
    - value = 2

    Isso faz df[column] = df[column] * value
    """
    return apply_constant_to_col(column, value, 'prod', runtime)


@tool
@track_history(tool_type=ToolType.TRANSFORM_COL, modified_col='column')
def power_value(
    column: str,
    value: Union[int, float],
    runtime: ToolRuntime,
):
    """
    Eleva os valores de uma coluna da base por um valor fixo
    Exemplo:
    - column = 'idade'
    - value = 2

    Isso faz df[column] = df[column] ** value
    """
    return apply_constant_to_col(column, value, 'pow', runtime)


def apply_constant_to_col(
    column: str,
    value: Union[int, float],
    op_type: str,
    runtime: ToolRuntime,
) -> Command:
    """
    Retorna um comando para atualizar o dataset com a coluna transformada. Se falhar, a conversão 
    não é realizada
    """
    df: pd.DataFrame = runtime.state["dataset"]
    raise_if_column_not_in_dataset(df, column)
    raise_if_column_should_be_frozen(column, runtime.state['frozen_columns'])

    return_msg = None

    op_to_func = {
        'subtraction': (lambda x: x[column] - value),
        'sum': (lambda x: x[column] + value),
        'div': (lambda x: x[column] / value),
        'prod': (lambda x: x[column] * value),
        'pow': (lambda x: x[column]**value)
    }

    try:
        df[column] = op_to_func[op_type](df)
        return_msg = f"Foi aplicado a operação de {op_type} na coluna {column} com o valor {value} com sucesso."
    except Exception as e:
        raise ValueError(
            f"Não foi possível realizar a operação {op_type} usando o valor {value} na coluna '{column}'! Erro: {str(e)}"
        )
    else:
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
@track_history(tool_type=ToolType.TRANSFORM_COL, modified_col='column')
def map_col_values(column: str, mapping: Dict[Any, Any], runtime: ToolRuntime):
    """
    Mapeia valores da coluna
    Exemplo:
    - column: classe
    - mapping: {'classe_a':1, 'classe_b':0}
    Isso faz: df[column] = df[column].map(mapping)
    """
    df: pd.DataFrame = runtime.state["dataset"]
    raise_if_column_not_in_dataset(df, column)
    raise_if_column_should_be_frozen(column, runtime.state['frozen_columns'])

    df[column] = df[column].map(
        mapping,
        na_action='ignore',
    )
    return_msg = f"A coluna '{column}' foi mapeada com sucesso."

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
