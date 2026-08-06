import pandas as pd

from src_data_readyness_agent.common.data_structs import ActionType, Severity
from src_data_readyness_agent.testing import expectations_structs, test_dataset


def get_dataset_config() -> test_dataset.DatasetConfig:
    return test_dataset.DatasetConfig(target_col='faturamento',
                                      prefered_language='Português',
                                      eval_agent_max_its=5,
                                      eval_agent_max_supersteps=45,
                                      transform_agent_max_its=15,
                                      transform_agent_max_supersteps=45,
                                      user_entry="")


def get_transform_expectations(
) -> expectations_structs.TransformAgentExpectations:
    expected_dtypes_per_col = {
        'metragem': 'numeric',
        'classe': 'numeric',
        'especial': 'numeric',
        'data_abertura': 'datetime',
        'faturamento': 'numeric',
        'qt_mesas': 'numeric',
        'Loja': 'string'
    }

    cols_expected_to_not_have_nulls = ('metragem', 'especial')

    cols_expected_to_have_cardinality = {'especial': 3, 'classe': 4}

    cols_expected_num_range = {
        'metragem': {
            'min': 43,
            'max': 62
        },
        'qt_mesas': {
            'min': 4,
            'max': 7
        }
    }

    return expectations_structs.TransformAgentExpectations(
        expected_dtypes_per_col, cols_expected_to_not_have_nulls,
        cols_expected_to_have_cardinality, cols_expected_num_range)


def get_eval_expectations() -> expectations_structs.EvalAgentExpectations:
    expected_actions_per_col = {
        'metragem': [ActionType.FILL_MISSING_VALUES],
        'classe': [ActionType.ENCODE_CATEGORICAL],
        'especial':
        [ActionType.FILL_MISSING_VALUES, ActionType.ENCODE_CATEGORICAL],
        'data_abertura': [ActionType.CONVERT_DTYPE]
    }

    expected_severity_per_col = {
        'metragem': Severity.MEDIUM,
        'classe': Severity.LOW,
        'especial': Severity.MEDIUM,
        'data_abertura': Severity.MEDIUM,
        'faturamento': Severity.NO_PROBLEM,
        'qt_mesas': Severity.NO_PROBLEM,
        'Loja': Severity.NO_PROBLEM
    }

    return expectations_structs.EvalAgentExpectations(
        expected_actions_per_col, expected_severity_per_col)


if __name__ == "__main__":
    dataset = pd.read_csv("data/test_data.csv")
    OPENAI_API_KEY = ""
    LANGSMITH_API_KEY = ""
    LANGSMITH_PROJECT = "data_readiness_agent"

    # Expectativas do agente de avaliação
    eval_expectations = get_eval_expectations()

    # Expectativas do agente de transformação
    transform_expectations = get_transform_expectations()

    dataset_config = get_dataset_config()
    #TODO: COLETAR AS ESTATÍSTICAS DE USO DE TOOLS
    #TODO: CRIAR UM DATACLASS COM O RESULTADO DO TESTE ENGLOBANDO TUDO QUE É RETORNADO
    #TODO: COMPLETAR AS CONFIGURAÇÕES DE MODELOS A SEREM ANALISADAS
    eval_results, transform_results = test_dataset.main(
        dataset,
        OPENAI_API_KEY,
        LANGSMITH_API_KEY,
        LANGSMITH_PROJECT,
        eval_expectations,
        transform_expectations,
        dataset_config,
        runs_per_config=4,
    )

    eval_results.to_csv(
        "src_data_readyness_agent/testing/test_results/eval_agent_dummy_data.csv",
        index=None)
    transform_results.to_csv(
        "src_data_readyness_agent/testing/test_results/transformation_agent_dummy_data.csv",
        index=None)
