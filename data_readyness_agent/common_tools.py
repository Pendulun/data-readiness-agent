import pandas as pd
from langchain.tools import tool, ToolRuntime
from typing_extensions import Any, Dict, List

# Tools que acessam o dataset_profile


@tool
def get_n_rows(runtime: ToolRuntime) -> int:
    """Retorna a quantidade de linhas na base do profile original"""
    return runtime.state['dataset_profile'].n_rows


@tool
def get_n_cols(runtime: ToolRuntime) -> int:
    """Retorna a quantidade de colunas na base a partir do profile original"""
    return runtime.state['dataset_profile'].n_columns


@tool
def get_null_counts(runtime: ToolRuntime) -> int:
    """Retorna a quantidade de valores nulos de cada coluna na base a partir do profile original"""
    return runtime.state['dataset_profile'].null_counts


@tool
def get_unique_counts(runtime: ToolRuntime) -> int:
    """Retorna a quantidade de valores únicos em cada coluna na base a partir do profile original"""
    return runtime.state['dataset_profile'].unique_counts


@tool
def get_col_preview(column: str,
                    runtime: ToolRuntime) -> Dict[str, List[Any] | str]:
    """Retorna um preview dos valores contidos na coluna da base a partir do profile original"""
    data_profile = runtime.state['dataset_profile']
    if column not in data_profile.samples.keys():
        return {"error": f"Coluna {column} não está presente na base"}

    return data_profile.samples[column]
