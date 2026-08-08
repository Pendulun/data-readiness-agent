import pandas as pd

from src_data_readyness_agent.common.data_structs import ActionType, Severity
from src_data_readyness_agent.testing import expectations_structs, config, test_dataset


def get_dataset_config() -> test_dataset.DatasetConfig:
    return test_dataset.DatasetConfig(target_col='Did Friend 5 sleep',
                                      prefered_language='Português',
                                      eval_agent_max_its=20,
                                      eval_agent_max_supersteps=80,
                                      transform_agent_max_its=35,
                                      transform_agent_max_supersteps=105,
                                      user_entry="")


def get_transform_expectations(
) -> expectations_structs.TransformAgentExpectations:
    expected_numeric_cols = ('Year', 'Type 1', 'Type 2', 'Animation',
                             'Origin Country', 'Friend Who chose it',
                             'Did Friend 5 sleep', 'Duration (minutes)',
                             'Friend 2 score', 'Friend 5 score',
                             'Friend 7 score', 'Friend 8 score', 'Mean score',
                             'IMDB score', 'IMDB-Mean')
    expected_dtypes_per_col = {col: 'numeric' for col in expected_numeric_cols}
    expected_dtypes_per_col['Movie'] = 'string'
    expected_dtypes_per_col['Watch Date'] = 'datetime'

    cols_expected_to_not_have_nulls = ('Year', 'Type 1', 'Type 2', 'Animation',
                                       'Origin Country', 'Friend Who chose it',
                                       'Did Friend 5 sleep',
                                       'Duration (minutes)', 'Friend 2 score',
                                       'Friend 5 score', 'Friend 7 score',
                                       'Friend 8 score', 'Mean score',
                                       'IMDB score', 'IMDB-Mean', 'Movie',
                                       'Watch Date')

    cols_expected_to_have_cardinality = {
        'Origin Country': 11,
    }

    cols_expected_num_range = {
        'Year': {
            'min': 1975,
            'max': 2026
        },
        'Animation': {
            'min': 0,
            'max': 1
        },
        'Friend Who chose it': {
            'min': 1,
            'max': 8
        },
        'Duration (minutes)': {
            'min': 66,
            'max': 176
        },
        'Mean score': {
            'min': 0,
            'max': 10
        },
        'IMDB score': {
            'min': 0,
            'max': 10
        },
    }

    return expectations_structs.TransformAgentExpectations(
        expected_dtypes_per_col, cols_expected_to_not_have_nulls,
        cols_expected_to_have_cardinality, cols_expected_num_range)


def get_eval_expectations() -> expectations_structs.EvalAgentExpectations:
    expected_actions_per_col = {
        'ID': [ActionType.DROP_COLUMN],
        'Watch Date': [ActionType.CONVERT_DTYPE],
        'Year': [ActionType.CREATE_DERIVED_COL],
        'Type 1': [ActionType.ENCODE_CATEGORICAL],
        'Type 2':
        [ActionType.ENCODE_CATEGORICAL, ActionType.FILL_MISSING_VALUES],
        'Origin Country': [ActionType.ENCODE_CATEGORICAL],
        'Mean score': [ActionType.CONVERT_DTYPE],
        'IMDB score': [ActionType.CONVERT_DTYPE],
        'IMDB-Mean': [ActionType.CONVERT_DTYPE]
    }

    expected_severity_per_col = {
        'ID': Severity.LOW,
        'Movie': Severity.NO_PROBLEM,
        'Watch Date': Severity.MEDIUM,
        'Year': Severity.LOW,
        'Type 1': Severity.MEDIUM,
        'Type 2': Severity.MEDIUM,
        'Animation': Severity.NO_PROBLEM,
        'Origin Country': Severity.MEDIUM,
        'Mean score': Severity.HIGH,
        'IMDB score': Severity.HIGH,
        'IMDB-Mean': Severity.HIGH,
        'Did Friend 5 sleep': Severity.MEDIUM,
        'Duration (minutes)': Severity.NO_PROBLEM,
        'Friend 2 score': Severity.LOW,
        'Friend 5 score': Severity.LOW,
        'Friend 7 score': Severity.LOW,
        'Friend 8 score': Severity.LOW,
        'Animation': Severity.NO_PROBLEM,
    }

    return expectations_structs.EvalAgentExpectations(
        expected_actions_per_col, expected_severity_per_col)


if __name__ == "__main__":
    dataset = pd.read_csv(config.TEST_DATA_3_PATH)
    OPENAI_API_KEY = ""
    LANGSMITH_API_KEY = ""
    LANGSMITH_PROJECT = "data_readiness_agent"

    # Expectativas do agente de avaliação
    eval_expectations = get_eval_expectations()

    # Expectativas do agente de transformação
    transform_expectations = get_transform_expectations()

    dataset_config = get_dataset_config()
    results: test_dataset.BenchmarkResults = test_dataset.main(
        dataset,
        OPENAI_API_KEY,
        LANGSMITH_API_KEY,
        LANGSMITH_PROJECT,
        eval_expectations,
        transform_expectations,
        dataset_config,
        runs_per_config=4,
    )

    results.eval_agent_results.to_csv(
        "src_data_readyness_agent/testing/test_results/eval_agent_movie_ratings_data.csv",
        index=None)
    results.transform_agent_results.to_csv(
        "src_data_readyness_agent/testing/test_results/transformation_agent_movie_ratings_data.csv",
        index=None)
    results.transform_agent_tool_usage_results.to_csv(
        "src_data_readyness_agent/testing/test_results/transformation_agent_tool_usage_movie_ratings_data.csv",
        index=None)
