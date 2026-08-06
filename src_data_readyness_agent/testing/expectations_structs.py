from dataclasses import dataclass
from typing_extensions import Dict, List, Tuple

from src_data_readyness_agent.common.data_structs import ActionType, Severity


@dataclass
class EvalAgentExpectations():
    expected_actions_per_col: Dict[str, List[ActionType]]
    expected_severity_per_col: Dict[str, Severity]


@dataclass
class TransformAgentExpectations():
    expected_dtypes_per_col: Dict[str, str]
    cols_expected_to_not_have_nulls: Tuple[str]
    cols_expected_to_have_cardinality: Dict[str, int]
    cols_expected_num_range: Dict[str, Dict[str, int]]
