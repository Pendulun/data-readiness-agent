from collections.abc import Callable
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest

from src_data_readyness_agent.data_transformation_agent.data_structs import ToolHistory
from src_data_readyness_agent.data_transformation_agent.utils import ToolType


@wrap_tool_call
def add_to_tool_history(request: ToolCallRequest,
                        handler: Callable[[ToolCallRequest], ToolMessage]):
    tool_name = request.tool.name
    args = request.tool_call["args"]

    try:
        result = handler(request)

        tool_history: ToolHistory = request.state["tool_history"]
        tool_history.add_usage(tool_name=tool_name,
                               args_dict=args,
                               sucess=True)

        if getattr(request.tool.func, "tool_type",
                   None) in ToolType.tools_that_modify_columns():
            target_col = args[getattr(request.tool.func, "modified_col")]
            tool_history.add_col_usage(column=target_col,
                                       tool_name=tool_name,
                                       args_dict=args,
                                       sucess=True)

        return result

    except Exception:
        request.state["tool_history"].add_usage(tool_name=tool_name,
                                                args_dict=args,
                                                sucess=False)
        # Isso deixa o langchain resolver a exceção dependendo se a tool está indicada
        # com handle_tool_errors=Tool na sua definição
        raise


@wrap_tool_call
def handle_tool_errors(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage],
) -> ToolMessage:
    """
    Transforma o raise de uma tool em uma ToolMessage
    """
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=str(e),
            tool_call_id=request.tool_call["id"],
        )
