import pandas as pd
from langchain.tools import tool, ToolRuntime
from typing_extensions import Any, Dict, List

# Tools do agente de avaliação da base


@tool
def get_n_rows(runtime: ToolRuntime) -> int:
    """Retorna a quantidade de linhas na base"""
    return runtime.state['dataset_profile'].n_rows


@tool
def get_n_cols(runtime: ToolRuntime) -> int:
    """Retorna a quantidade de colunas na base"""
    return runtime.state['dataset_profile'].n_columns


@tool
def get_null_counts(runtime: ToolRuntime) -> int:
    """Retorna a quantidade de valores nulos de cada coluna na base"""
    return runtime.state['dataset_profile'].null_counts


@tool
def get_unique_counts(runtime: ToolRuntime) -> int:
    """Retorna a quantidade de valores únicos em cada coluna na base"""
    return runtime.state['dataset_profile'].unique_counts


@tool
def get_col_preview(column: str,
                    runtime: ToolRuntime) -> Dict[str, List[Any] | str]:
    """Retorna um preview dos valores contidos na coluna da base"""
    data_profile = runtime.state['dataset_profile']
    if column not in data_profile.samples.keys():
        return {"error": f"Coluna {column} não está presente na base"}

    return data_profile.samples[column]


@tool
def check_duplicate_rows(columns: List[str],
                         runtime: ToolRuntime) -> Dict[str, int]:
    """Retorna a quantidade de linhas duplicadas usando o subconjunto de colunas informadas"""
    df = runtime.state['dataset']
    target_cols = list(set([col for col in columns if col in df.columns]))
    if not target_cols:
        return {
            "error": ("Nenhuma das colunas informadas existe na base.",
                      "Colunas solicitadas: " + str(columns),
                      "Colunas disponíveis: " + str(df.columns.tolist()))
        }

    qt_duplicados = int(df.duplicated(subset=target_cols).sum())
    return {'qt_duplicados': qt_duplicados}


# Essa função existe para evitar de o agente chamar a função check_duplicate_rows
# informando todas as colunas existentes
@tool
def check_duplicate_rows_all_cols(runtime: ToolRuntime) -> Dict[str, int]:
    """Retorna a quantidade de linhas duplicadas usando todas as colunas da base"""
    df = runtime.state['dataset']

    qt_duplicados = int(df.duplicated().sum())
    return {'qt_duplicados': qt_duplicados}


@tool
def check_column_consistency(col_name: str, runtime: ToolRuntime) -> dict:
    """Retorna quantos tipos de dados diferentes a coluna informada possui"""
    df = runtime.state['dataset']

    if col_name not in df.columns:
        return {"error": f"A coluna '{col_name}' não existe na base."}

    # Tipos Python encontrados
    value_types = (df[col_name].dropna().map(
        lambda x: type(x).__name__).value_counts().to_dict())

    return value_types


@tool
def get_column_value_distribution(col_name: str, runtime: ToolRuntime) -> dict:
    """Retorna a distribuição de até 50 valores mais comuns da coluna informada"""
    df = runtime.state['dataset']

    if col_name not in df.columns:
        return {"error": f"A coluna '{col_name}' não existe."}

    counts = (df[col_name].value_counts(dropna=False).head(50).to_dict())

    return {str(value): int(count) for value, count in counts.items()}


@tool
def analyze_missingness_patterns(col_name: str, runtime: ToolRuntime) -> dict:
    """
    Analisa se valores ausentes da coluna informada estão associados a outras colunas categóricas.
    Só chame para colunas que, de fato, possuam valores nulos.
    """
    df = runtime.state['dataset']

    if col_name not in df.columns:
        return {"error": f"Coluna '{col_name}' não encontrada."}

    missing_mask = df[col_name].isna()

    if missing_mask.sum() == 0:
        return {"msg": "A coluna não possui valores faltantes."}

    results = {}

    for col in df.columns:
        if col == col_name:
            continue

        # Se for uma coluna categórica
        if df[col].dtype == "object":
            # Conta a porcentagem de valores em col em que col_name é nulo
            grouped = (missing_mask.groupby(
                df[col]).mean().sort_values(ascending=False))

            results[col] = {
                str(k): round(float(v), 2)
                for k, v in grouped.items()
            }

    return results


@tool
def detect_outliers(col_name: str, runtime: ToolRuntime) -> dict:
    """
    Detecta possíveis outliers na coluna numérica informada usando o método IQR.

    Retorna a quantidade, o percentual de outliers e os limites
    inferior e superior para a detecção.
    """
    df = runtime.state['dataset']

    if col_name not in df.columns:
        return {"error": f"A coluna '{col_name}' não existe na base."}

    col = df[col_name]

    # Verifica se a coluna é numérica
    if not pd.api.types.is_numeric_dtype(col):
        return {
            "message": "A coluna não é numérica. Não é possível aplicar IQR."
        }

    # Remove valores nulos para calcular os quartis
    values = col.dropna()

    if len(values) == 0:
        return {"message": "A coluna não possui valores válidos."}

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    n_lower_outliers = (col < lower_bound).sum()
    n_upper_outliers = (col > upper_bound).sum()

    n_outliers = n_lower_outliers + n_upper_outliers
    n_valid = int(values.shape[0])

    return {
        "q1":
        round(float(q1), 2),
        "q3":
        round(float(q3), 2),
        "iqr":
        round(float(iqr), 2),
        "lower_bound":
        round(float(lower_bound), 2),
        "upper_bound":
        round(float(upper_bound), 2),
        "n_upper_outliers":
        n_upper_outliers,
        "n_lower_outliers":
        n_lower_outliers,
        "outlier_percentage":
        round((n_outliers / n_valid * 100 if n_valid > 0 else 0), 2)
    }


@tool
def get_columns_names(runtime: ToolRuntime) -> List[str]:
    """Retorna os nomes de todas as colunas existentes na base"""
    df = runtime.state['dataset']
    return df.columns.tolist()
