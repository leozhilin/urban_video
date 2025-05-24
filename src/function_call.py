# encoding: utf-8
import json
from src.tools.video_summary import video_summary

class FunctionCall:
    def __init__(self):
        self.tool_functions = {
            "video_summary": video_summary,
        }

    async def handle_tool_call(self, function_call_info):
        """
        处理单个工具调用并返回结果消息

        Args:
            function_call_info: 模型返回的函数调用信息字典

        Returns:
            函数调用响应消息
        """
        print(f"function_call_info: {function_call_info}\n\n")
        name = function_call_info.get("name")
        args_str = function_call_info.get("arguments", "{}")

        # 获取对应的工具函数
        tool_function = self.tool_functions.get(name)

        if not tool_function:
            result = {
                "output": "error",
                "error": f"未找到名为'{name}'的函数"
            }
        else:
            try:
                # 将args字符串解析为字典
                args = json.loads(args_str)
                print(f"args: {args}\n\n")
                # 调用工具函数
                result = await tool_function(**args)

                # 如果结果是字符串，尝试解析为JSON
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except json.JSONDecodeError:
                        result = {"output": result}

                # 确保有个output字段
                if "output" not in result:
                    result["output"] = "done"

            except Exception as e:
                import traceback
                result = {
                    "output": "error",
                    "error": f"函数调用失败: {str(e)}",
                    "traceback": traceback.format_exc()
                }

        # 构建函数响应消息
        function_response_message = {
            "role": "tool",
            "content": str(result),
            "tool_call_id": function_call_info.get("id")
        }

        return function_response_message
