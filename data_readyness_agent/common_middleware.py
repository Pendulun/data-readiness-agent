from langchain.agents.middleware import AgentMiddleware
from langchain.messages import AIMessage


class DebugMiddleware(AgentMiddleware):

    def after_model(self, state, runtime):
        last_message = state["messages"][-1]

        if isinstance(last_message, AIMessage):
            print("\n--- LLM RESPONSE ---")
            print("Content:", last_message.content)
            print("Tool calls:", last_message.tool_calls)

        return None
