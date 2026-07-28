import copy
import numpy as np
import pandas as pd
from langchain.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from typing_extensions import Any, Dict, List, Union


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
def get_col_unique(column: str, runtime: ToolRuntime) -> List[Any]:
    """Retorna todos os valores únicos contidos em uma coluna alvo da base"""
    if column not in runtime.state['dataset'].columns.tolist():
        return {"error": f"Coluna {column} não está presente na base"}
    serie: pd.Series = runtime.state['dataset'][column]
    return serie.unique().tolist()


@tool
def rename_column(column: str, new_column: str, runtime: ToolRuntime):
    """
    Renomeia uma coluna na base.
    """
    df: pd.DataFrame = runtime.state["dataset"]
    if column not in runtime.state['dataset'].columns.tolist():
        return {"error": f"Coluna {column} não está presente na base"}

    if new_column in runtime.state['dataset'].columns.tolist():
        return {"error": f"Coluna {column} já está presente na base"}

    df.rename(columns={column: new_column}, inplace=True)
    new_tool_history: dict = copy.deepcopy(runtime.state['tool_history'])
    new_tool_history.setdefault(column, list()).append({
        'tool': 'rename_column',
        'args': {
            'new_column': new_column
        }
    })

    return_msg = f"Coluna {column} renomeada com sucesso para {new_column}"

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
    return get_command_with_converted_column_type(column, 'datetime', runtime)


@tool
def convert_column_to_int(
    column: str,
    runtime: ToolRuntime,
):
    """Converte uma coluna para o tipo inteiro"""
    return get_command_with_converted_column_type(column, 'int', runtime)


@tool
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
    if column not in df.columns.tolist():
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(f"Coluna {column} não está presente na base"),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            })

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


@tool
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
    error_msg = None
    for col in [first_col, second_col]:
        if col not in df.columns.tolist():
            error_msg = f"Coluna {col} não está presente na base"
            break

    if error_msg is None and new_col_name in df.columns.tolist():
        error_msg = f"Coluna {new_col_name} já está presente na base"

    if error_msg is not None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(error_msg),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            })

    return_msg = None

    op_to_func = {
        'subtraction': (lambda x: x[first_col] - x[second_col]),
        'sum': (lambda x: x[first_col] + x[second_col]),
        'div': (lambda x: x[first_col] / np.maximum(x[second_col], 1)),
        'prod': (lambda x: x[first_col] * x[second_col]),
        'pow': (lambda x: x[first_col]**x[second_col])
    }
    new_tool_history: dict = copy.deepcopy(runtime.state['tool_history'])

    try:
        df[new_col_name] = op_to_func[op_type](df)
        return_msg = f"Os valores das colunas '{first_col}' e '{second_col}' foram usados para criar {new_col_name} com sucesso."
    except Exception as e:
        return_msg = f"Não foi possível realizar a operação {op_type} usando valores das coluna '{first_col}' e {second_col}! Erro: {str(e)}"
    else:
        new_tool_history.setdefault(new_col_name, list()).append({
            'tool': f'{op_type}_cols',
            'args': {
                'first_col': first_col,
                'second_col': second_col
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
    error_msg = None
    if column not in df.columns.tolist():
        error_msg = f"Coluna {column} não está presente na base"
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(error_msg),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            })

    return_msg = None

    op_to_func = {
        'subtraction': (lambda x: x[column] - value),
        'sum': (lambda x: x[column] + value),
        'div': (lambda x: x[column] / value),
        'prod': (lambda x: x[column] * value),
        'pow': (lambda x: x[column]**value)
    }
    new_tool_history: dict = copy.deepcopy(runtime.state['tool_history'])

    try:
        df[column] = op_to_func[op_type](df)
        return_msg = f"Foi aplicado a operação de {op_type} na coluna {column} com o valor {value} com sucesso."
    except Exception as e:
        return_msg = f"Não foi possível realizar a operação {op_type} usando o valor {value} na coluna '{column}'! Erro: {str(e)}"
    else:
        new_tool_history.setdefault(column, list()).append({
            'tool': f'{op_type}_value',
            'args': {
                'value': value,
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
def map_col_values(column: str, mapping: Dict[Any, Any], runtime: ToolRuntime):
    """
    Mapeia valores da coluna
    Exemplo:
    - column: classe
    - mapping: {'classe_a':1, 'classe_b':0}
    Isso faz: df[column] = df[column].map(mapping)
    """
    df: pd.DataFrame = runtime.state["dataset"]
    if column not in runtime.state['dataset'].columns.tolist():
        return {"error": f"Coluna {column} não está presente na base"}

    df[column] = df[column].map(
        mapping,
        na_action='ignore',
    )
    return_msg = f"A coluna '{column}' foi mapeada com sucesso."
    new_tool_history: dict = copy.deepcopy(runtime.state['tool_history'])
    new_tool_history.setdefault(column, list()).append({
        'tool': 'map_col_values',
        'args': {
            'mapping': mapping
        }
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
def copy_column(column: str, new_column: str, runtime: ToolRuntime):
    """
    Cria uma cópia de uma coluna na base para uma nova coluna
    Exemplo: df[new_column] = df[column].values
    """
    df: pd.DataFrame = runtime.state["dataset"]
    if column not in runtime.state['dataset'].columns.tolist():
        return {"error": f"Coluna {column} não está presente na base"}

    if new_column in runtime.state['dataset'].columns.tolist():
        return {"error": f"Coluna {column} já está presente na base"}

    df[new_column] = df[column].values
    return_msg = f"A coluna '{column}' foi copiada para '{new_column}' com sucesso."
    new_tool_history: dict = copy.deepcopy(runtime.state['tool_history'])
    new_tool_history.setdefault(column, list()).append({
        'tool': 'copy_column',
        'args': {
            'new_column': new_column
        }
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
