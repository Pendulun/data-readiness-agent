import pandas as pd
from typing_extensions import Tuple

from src_data_readyness_agent.common.data_structs import ActionType, Action, ColumnIssue, EvalAgentResponse, ReadinessStatus, Severity
from src_data_readyness_agent.data_transformation_agent.data_structs import ToolHistory


def create_dummy_avaliacao() -> EvalAgentResponse:
    dummy_findings = [
        ColumnIssue(
            column='metragem',
            severity=Severity.MEDIUM,
            suggested_actions=[
                Action(explanation="A",
                       recommended_action_type=ActionType.FILL_MISSING_VALUES)
            ],
        ),
        ColumnIssue(
            column='classe',
            severity=Severity.MEDIUM,
            suggested_actions=[
                Action(explanation="A",
                       recommended_action_type=ActionType.ENCODE_CATEGORICAL)
            ],
        ),
        ColumnIssue(
            column='especial',
            severity=Severity.MEDIUM,
            suggested_actions=[
                Action(explanation="A",
                       recommended_action_type=ActionType.FILL_MISSING_VALUES),
                Action(explanation="A",
                       recommended_action_type=ActionType.ENCODE_CATEGORICAL),
            ],
        ),
        ColumnIssue(
            column='data_abertura',
            severity=Severity.MEDIUM,
            suggested_actions=[
                Action(explanation="A",
                       recommended_action_type=ActionType.CONVERT_DTYPE)
            ],
        ),
    ]
    avaliacao = EvalAgentResponse(readiness_status=ReadinessStatus.READY,
                                  summary="Teste",
                                  findings=dummy_findings)
    return avaliacao


def create_dummy_transformed_data_and_tool_history(
) -> Tuple[pd.DataFrame, ToolHistory]:
    dados_transformados = pd.read_csv(
        "src_data_readyness_agent/testing/dummy_dummy_dataset_transformed.csv")
    tool_history = ToolHistory()

    tools = [{
        'name': 'rename_column',
        'args': {
            'column': 'metragem',
            'new_column': 'nova_metragem'
        },
        'target_col': 'column'
    }]

    for tool in tools:
        tool_history.add_usage(tool_name=tool['name'],
                               args_dict=tool['args'],
                               sucess=True)

        tool_history.add_col_usage(column=tool['args'][tool['target_col']],
                                   tool_name=tool['name'],
                                   args_dict=tool['args'],
                                   sucess=True)

    return dados_transformados, tool_history
