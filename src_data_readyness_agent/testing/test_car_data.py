import pandas as pd

from src_data_readyness_agent.common.data_structs import ActionType, Severity
from src_data_readyness_agent.testing import expectations_structs, config, test_dataset


def get_dataset_config() -> test_dataset.DatasetConfig:
    return test_dataset.DatasetConfig(target_col='car_prices_in_rupee',
                                      prefered_language='Português',
                                      eval_agent_max_its=15,
                                      eval_agent_max_supersteps=105,
                                      transform_agent_max_its=35,
                                      transform_agent_max_supersteps=105,
                                      user_entry="")


def get_transform_expectations(
) -> expectations_structs.TransformAgentExpectations:
    expected_dtypes_per_col = {
        'car_name': 'string',
        'car_prices_in_rupee': 'numeric',
        'kms_driven': 'numeric',
        'fuel_type': 'numeric',
        'transmission': 'numeric',
        'ownership': 'numeric',
        'manufacture': 'numeric',
        'engine': 'numeric',
        'Seats': 'numeric'
    }

    cols_expected_to_not_have_nulls = ('car_prices_in_rupee', 'kms_driven',
                                       'fuel_type', 'transmission',
                                       'ownership', 'manufacture', 'engine',
                                       'Seats')

    cols_expected_to_have_cardinality = {
        'fuel_type': 5,
        'transmission': 2,
        'ownership': 6,
    }

    cols_expected_num_range = {
        'car_prices_in_rupee': {
            'min': 35_000,
            'max': 19_200_000
        },
        'kms_driven': {
            'min': 250,
            'max': 560_000
        },
        'engine': {
            'min': 0,
            'max': 5950
        },
        'Seats': {
            'min': 2,
            'max': 8
        },
    }

    return expectations_structs.TransformAgentExpectations(
        expected_dtypes_per_col, cols_expected_to_not_have_nulls,
        cols_expected_to_have_cardinality, cols_expected_num_range)


def get_eval_expectations() -> expectations_structs.EvalAgentExpectations:
    expected_actions_per_col = {
        'Unnamed: 0': [ActionType.DROP_COLUMN],
        'car_prices_in_rupee': [ActionType.CONVERT_DTYPE],
        'kms_driven': [ActionType.CONVERT_DTYPE],
        'fuel_type': [ActionType.ENCODE_CATEGORICAL],
        'transmission': [ActionType.ENCODE_CATEGORICAL],
        'ownership': [ActionType.ENCODE_CATEGORICAL],
        'engine': [ActionType.CONVERT_DTYPE],
        'Seats': [ActionType.CONVERT_DTYPE],
    }

    expected_severity_per_col = {
        'Unnamed: 0': Severity.LOW,
        'car_name': Severity.NO_PROBLEM,
        'car_prices_in_rupee': Severity.HIGH,
        'kms_driven': Severity.HIGH,
        'fuel_type': Severity.MEDIUM,
        'transmission': Severity.MEDIUM,
        'ownership': Severity.MEDIUM,
        'manufacture': Severity.NO_PROBLEM,
        'engine': Severity.HIGH,
        'Seats': Severity.HIGH
    }

    return expectations_structs.EvalAgentExpectations(
        expected_actions_per_col, expected_severity_per_col)


if __name__ == "__main__":
    dataset = pd.read_csv(config.TEST_DATA_2_PATH)
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
        "src_data_readyness_agent/testing/test_results/eval_agent_car_price_data.csv",
        index=None)
    results.transform_agent_results.to_csv(
        "src_data_readyness_agent/testing/test_results/transformation_agent_car_price_data.csv",
        index=None)
    results.transform_agent_tool_usage_results.to_csv(
        "src_data_readyness_agent/testing/test_results/transformation_agent_tool_usage_car_price_data.csv",
        index=None)
