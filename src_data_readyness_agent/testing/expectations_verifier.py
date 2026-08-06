import pandas as pd
from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_object_dtype,
    is_string_dtype,
    is_datetime64_any_dtype,
)
from typing_extensions import Dict, List, Tuple

from src_data_readyness_agent.common.data_structs import ActionType, ColumnIssue, Severity
from src_data_readyness_agent.data_transformation_agent.data_structs import ToolHistory, ToolHistoryUsage, ToolUsageInfo, ToolUsagePerCol, SingleToolCall
from src_data_readyness_agent.testing import expectations_structs


def get_eval_agent_verification_results(
    expectations: expectations_structs.EvalAgentExpectations,
    findings: List[ColumnIssue],
) -> pd.DataFrame:
    """
    Constrói a tabela com os as verificações para o agente de avaliação
    """
    expected_actions_results = _verify_expected_actions(
        expectations.expected_actions_per_col, findings)
    expected_severities_results = _verify_expected_severities(
        expectations.expected_severity_per_col, findings)

    final_result = dict()
    for col, col_results in expected_actions_results.items():
        for action_type, result in col_results.items():
            final_result[col + "_expected_action_has_" +
                         action_type] = [result]

    for col, result in expected_severities_results.items():
        final_result[col + "_severity_is_" +
                     expectations.expected_severity_per_col[col].value] = [
                         result
                     ]

    return pd.DataFrame(final_result)


def _verify_expected_actions(
    expected_actions_per_col: Dict[str, List[ActionType]],
    findings: List[ColumnIssue],
) -> Dict[str, Dict[str, bool]]:
    """
    Verifica se os tipos de ações recomendadas para cada coluna estão de acordo
    com as ações esperadas
    """
    results = {
        col: {
            action.value: False
            for action in actions
        }
        for col, actions in expected_actions_per_col.items()
    }
    for finding in findings:
        # Só checamos colunas nas quais realmente esperamos alguma coisa
        if finding.column in expected_actions_per_col:
            expected_col_actions = expected_actions_per_col[finding.column]
            for action in finding.suggested_actions:
                # Para cada tipo de ação recomendada, se ela realmente for esperada,
                # marcamos como atingida
                if action.recommended_action_type in expected_col_actions:
                    results[finding.column][
                        action.recommended_action_type.value] = True

    return results


def _verify_expected_severities(
    expected_severities_per_col: Dict[str, Severity],
    findings: List[ColumnIssue],
) -> Dict[str, bool]:
    """
    Verifica se o nível de severidade dos problemas encontrados estão de acordo com os 
    esperados para cada coluna
    """
    results = {col: False for col in expected_severities_per_col.keys()}
    for finding in findings:
        # Só checamos colunas nas quais realmente esperamos alguma coisa
        if finding.column in expected_severities_per_col:
            if finding.severity == expected_severities_per_col[finding.column]:
                results[finding.column] = True

    return results


def get_transform_agent_verification_results(
    dataset: pd.DataFrame, original_columns_current_names: Dict[str, str],
    transform_expectations: expectations_structs.TransformAgentExpectations
) -> pd.DataFrame:
    """
    Constrói a tabela com os as verificações para o agente de avaliação
    """
    expected_dtypes_results = _verify_expected_dtypes(
        dataset,
        original_columns_current_names,
        transform_expectations.expected_dtypes_per_col,
    )

    expected_nullity_results = _verify_nullity(
        dataset,
        original_columns_current_names,
        transform_expectations.cols_expected_to_not_have_nulls,
    )

    expected_cardinality_results = _verify_expected_cardinality(
        dataset,
        original_columns_current_names,
        transform_expectations.cols_expected_to_have_cardinality,
    )

    expected_num_range_results = _verify_expected_num_range(
        dataset,
        original_columns_current_names,
        transform_expectations.cols_expected_num_range,
    )

    final_result = dict()
    for col, col_result in expected_dtypes_results.items():
        final_result[col + "_dtype_is_" +
                     transform_expectations.expected_dtypes_per_col[col]] = [
                         col_result
                     ]

    for col, col_result in expected_nullity_results.items():
        final_result[col + "_doesnt_has_nulls"] = [col_result]

    for col, col_result in expected_cardinality_results.items():
        final_result[col + "_cardinality_is_" + str(
            transform_expectations.cols_expected_to_have_cardinality[col])] = [
                col_result
            ]

    for col, col_result in expected_num_range_results.items():
        final_result[col + "_is_between_" + str(
            transform_expectations.cols_expected_num_range[col]['min']) +
                     "_and_" +
                     str(transform_expectations.cols_expected_num_range[col]
                         ['max'])] = [col_result]

    return pd.DataFrame(final_result)


def _verify_expected_dtypes(
        dataset: pd.DataFrame, original_columns_current_names: Dict[str, str],
        expected_dtypes_per_col: Dict[str, str]) -> Dict[str, bool]:
    """
    Verifica o dtype das colunas de acordo com as expectativas
    """
    result = {col: False for col in expected_dtypes_per_col.keys()}
    for col, dtype in expected_dtypes_per_col.items():
        col_name = original_columns_current_names.get(col, col)
        if col_name in dataset.columns:
            result[col] = dtype == _get_dtype_category(dataset[col_name])
    return result


def _get_dtype_category(series: pd.Series) -> str:
    """
    Retorna o tipo de uma pd.Series
    """
    if is_numeric_dtype(series):
        return "numeric"
    elif is_bool_dtype(series):
        return "boolean"
    elif is_datetime64_any_dtype(series):
        return "datetime"
    elif is_string_dtype(series):
        return "string"
    elif is_object_dtype(series):
        return "object"
    else:
        return "other"


def _verify_nullity(
    dataset: pd.DataFrame,
    original_columns_current_names: Dict[str, str],
    cols_expected_to_not_have_nulls: Tuple[str],
) -> Dict[str, bool]:
    """
    Verifica se a nulidade das colunas informadas atingem as expectativas
    """
    result = {col: False for col in cols_expected_to_not_have_nulls}
    for col in cols_expected_to_not_have_nulls:
        col_name = original_columns_current_names.get(col, col)
        if col_name in dataset.columns:
            result[col] = not bool(dataset[col_name].isna().all())
    return result


def _verify_expected_cardinality(
    dataset: pd.DataFrame,
    original_columns_current_names: Dict[str, str],
    cols_expected_to_have_cardinality: Dict[str, int],
) -> Dict[str, bool]:
    """
    Verifica se a cardinalidade das colunas atingem as expectativas
    """
    result = {col: False for col in cols_expected_to_have_cardinality.keys()}
    for col, expected_card in cols_expected_to_have_cardinality.items():
        col_name = original_columns_current_names.get(col, col)
        if col_name in dataset.columns:
            result[col] = bool(dataset[col_name].nunique() == expected_card)
    return result


def _verify_expected_num_range(
    dataset: pd.DataFrame,
    original_columns_current_names: Dict[str, str],
    cols_expected_num_range: Dict[str, Dict[str, int]],
) -> Dict[str, bool]:
    """
    Verifica se o range numérico das colunas atingem as expectativas
    """
    result = {col: False for col in cols_expected_num_range.keys()}
    for col, exp_range in cols_expected_num_range.items():
        col_name = original_columns_current_names.get(col, col)
        if col_name in dataset.columns:
            try:
                result[col] = bool(
                    dataset[col_name].between(exp_range['min'],
                                              exp_range['max'],
                                              inclusive='both').all())
            except Exception as e:
                pass
    return result
