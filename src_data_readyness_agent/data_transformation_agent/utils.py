from enum import Enum
from typing_extensions import List


class ToolType(str, Enum):
    QUERY = "query"
    CREATE_COL = "create_col"
    DROP_COL = 'drop_col'
    TRANSFORM_COL = "transform_col"

    @classmethod
    def tools_that_modify_columns(cls) -> List['ToolType']:
        return [cls.CREATE_COL, cls.DROP_COL, cls.TRANSFORM_COL]


def track_history(tool_type: ToolType, modified_col: str = None):
    """
    Decorador que adiciona metadata em uma tool para que middlewares
    possam consumir isso
    """

    def decorator(func):
        func.tool_type = tool_type
        func.modified_col = modified_col
        return func

    return decorator
