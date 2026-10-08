import json

def get_sum(a, b):
    """
    Returns the sum of two numbers.
    
    Args:
        a (int or float): The first number.
        b (int or float): The second number.
    Returns:
        int or float: The sum of the two numbers.
    """
    return a + b

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_sum",
            "description": "Returns the sum of two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": ["number", "integer"],
                        "description": "The first number."
                    },
                    "b": {
                        "type": ["number", "integer"],
                        "description": "The second number."
                    }
                },
                " ": ["a", "b"]
            },
        },

    }
]

TOOL_NAMES = [tool["function"]["name"] for tool in TOOLS]
