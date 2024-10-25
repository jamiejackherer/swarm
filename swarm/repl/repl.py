import json
from typing import List, Optional

from swarm import Swarm
from swarm.types import Agent, Response


def _process_and_print_streaming_response(response: List[dict]) -> str:
    content: str = ""
    last_sender: str = ""

    for chunk in response:
        if "sender" in chunk:
            last_sender = chunk["sender"]

        if "content" in chunk and chunk["content"] is not None:
            if not content and last_sender:
                print(f"\033[94m{last_sender}:\033[0m", end=" ", flush=True)
                last_sender: str = ""
            print(chunk["content"], end="", flush=True)
            content += chunk["content"]

        if "tool_calls" in chunk and chunk["tool_calls"] is not None:
            for tool_call in chunk["tool_calls"]:
                f: dict = tool_call["function"]
                name: str = f["name"]
                if not name:
                    continue
                print(f"\033[94m{last_sender}: \033[95m{name}\033[0m()")

        if "delim" in chunk and chunk["delim"] == "end" and content:
            print()  # End of response message
            content = ""

        if "response" in chunk:
            return chunk["response"]


def _pretty_print_messages(messages: List[dict]) -> None:
    for message in messages:
        if message["role"] != "assistant":
            continue

        # print agent name in blue
        print(f"\033[94m{message['sender']}\033[0m:", end=" ")

        # print response, if any
        if message["content"]:
            print(message["content"])

        # print tool calls in purple, if any
        tool_calls: List[dict] = message.get("tool_calls") or []
        if len(tool_calls) > 1:
            print()
        for tool_call in tool_calls:
            f = tool_call["function"]
            name: str = f["name"]
            args: str = f["arguments"]
            arg_str: str = json.dumps(json.loads(args)).replace(":", "=")
            print(f"\033[95m{name}\033[0m({arg_str[1:-1]})")


def run_demo_loop(
    starting_agent: Agent,
    context_variables: Optional[dict] = None,
    stream: bool = False,
    debug: bool = False,
) -> None:
    client = Swarm()
    print("Starting Swarm CLI 🐝")

    messages: List[dict] = []
    agent: Agent = starting_agent

    while True:
        user_input: str = input("\033[90mUser\033[0m: ")
        messages.append({"role": "user", "content": user_input})

        response: Response = client.run(
            agent=agent,
            messages=messages,
            context_variables=context_variables or {},
            stream=stream,
            debug=debug,
        )

        if stream:
            response: str = _process_and_print_streaming_response(response)
        else:
            _pretty_print_messages(response.messages)

        messages.extend(response.messages)
        agent: Agent = response.agent
