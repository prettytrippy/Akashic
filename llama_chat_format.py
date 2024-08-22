from __future__ import annotations

import os
import json
import ctypes
import dataclasses
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Protocol

import llama_cpp.llama as llama
import llama_cpp.llama_types as llama_types
import llama_cpp.llama_grammar as llama_grammar

from ._utils import suppress_stdout_stderr


class LlamaChatCompletionHandler(Protocol):
    def __call__(
        self,
        *,
        llama: llama.Llama,
        messages: List[llama_types.ChatCompletionRequestMessage],
        functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
        function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
        tools: Optional[List[llama_types.ChatCompletionTool]] = None,
        tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        seed: Optional[int] = None,
        response_format: Optional[
            llama_types.ChatCompletionRequestResponseFormat
        ] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        logits_processor: Optional[llama.LogitsProcessorList] = None,
        grammar: Optional[llama.LlamaGrammar] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        **kwargs,  # type: ignore
    ) -> Union[
        llama_types.CreateChatCompletionResponse,
        Iterator[llama_types.CreateChatCompletionStreamResponse],
    ]:
        ...


CHAT_HANDLERS: Dict[str, LlamaChatCompletionHandler] = {}


def get_chat_completion_handler(name: str) -> LlamaChatCompletionHandler:
    return CHAT_HANDLERS[name]


def register_chat_completion_handler(name: str):
    def decorator(f: LlamaChatCompletionHandler):
        CHAT_HANDLERS[name] = f
        return f

    return decorator


def _get_system_message(
    messages: List[llama_types.ChatCompletionRequestMessage],
) -> str:
    """Get the first system message."""
    for message in messages:
        if message["role"] == "system":
            return message["content"] or ""
    return ""

def format_prompt(
    messages: List[llama_types.ChatCompletionRequestMessage],
    system_template: str,
    user: str,
    assistant: str,
    user_sep: str,
    assistant_sep: str,
    stop: str,
) -> ChatFormatterResponse:
    _system_message = _get_system_message(messages)
    system_message = system_template.format(system_message=_system_message) if system_template else ""
    # _stop = system_message if system_message else user
    _prompt = system_message
    for message in messages:
        role, content = message["role"], message["content"]
        if role == 'user':
            _prompt += user + content + user_sep
        elif role == 'assistant':
            _prompt += assistant + content + assistant_sep

    _prompt += assistant

    print("________\n", _prompt, "\n________", "\nSTOP:", stop)

    return ChatFormatterResponse(prompt=_prompt, stop=stop)


@dataclasses.dataclass
class ChatFormatterResponse:
    prompt: str
    stop: Optional[Union[str, List[str]]] = None


class ChatFormatter(Protocol):
    def __call__(
        self,
        *,
        messages: List[llama_types.ChatCompletionRequestMessage],
        **kwargs: Any,
    ) -> ChatFormatterResponse:
        ...


class BasicChatHandler:
    def __init__(self, chat_format: str):
        self.chat_format = chat_format


def _convert_text_completion_to_chat(
    completion: llama_types.Completion,
) -> llama_types.ChatCompletion:
    return {
        "id": "chat" + completion["id"],
        "object": "chat.completion",
        "created": completion["created"],
        "model": completion["model"],
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": completion["choices"][0]["text"],
                },
                "finish_reason": completion["choices"][0]["finish_reason"],
            }
        ],
        "usage": completion["usage"],
    }


def _convert_text_completion_chunks_to_chat(
    chunks: Iterator[llama_types.CreateCompletionStreamResponse],
) -> Iterator[llama_types.ChatCompletionChunk]:
    for i, chunk in enumerate(chunks):
        if i == 0:
            yield {
                "id": "chat" + chunk["id"],
                "model": chunk["model"],
                "created": chunk["created"],
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                        },
                        "finish_reason": None,
                    }
                ],
            }
        yield {
            "id": "chat" + chunk["id"],
            "model": chunk["model"],
            "created": chunk["created"],
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": chunk["choices"][0]["text"],
                    }
                    if chunk["choices"][0]["finish_reason"] is None
                    else {},
                    "finish_reason": chunk["choices"][0]["finish_reason"],
                }
            ],
        }


def _convert_completion_to_chat(
    completion_or_chunks: Union[
        llama_types.CreateCompletionResponse,
        Iterator[llama_types.CreateCompletionStreamResponse],
    ],
    stream: bool = False,
) -> Union[
    llama_types.CreateChatCompletionResponse, Iterator[llama_types.ChatCompletionChunk]
]:
    if stream:
        chunks: Iterator[llama_types.CreateCompletionStreamResponse] = completion_or_chunks  # type: ignore
        return _convert_text_completion_chunks_to_chat(chunks)
    else:
        completion: llama_types.Completion = completion_or_chunks  # type: ignore
        return _convert_text_completion_to_chat(completion)


_CHAT_FORMATS: Dict[str, ChatFormatter] = {}


def register_chat_format(name: str):
    def decorator(f: ChatFormatter):
        def basic_create_chat_completion(
            *,
            llama: llama.Llama,
            messages: List[llama_types.ChatCompletionRequestMessage],
            functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
            function_call: Optional[
                llama_types.ChatCompletionRequestFunctionCall
            ] = None,
            tools: Optional[List[llama_types.ChatCompletionTool]] = None,
            tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
            temperature: float = 0.2,
            top_p: float = 0.95,
            top_k: int = 40,
            min_p: float = 0.05,
            typical_p: float = 1.0,
            stream: bool = False,
            stop: Optional[Union[str, List[str]]] = [],
            seed: Optional[int] = None,
            response_format: Optional[
                llama_types.ChatCompletionRequestResponseFormat
            ] = None,
            max_tokens: Optional[int] = None,
            presence_penalty: float = 0.0,
            frequency_penalty: float = 0.0,
            repeat_penalty: float = 1.1,
            tfs_z: float = 1.0,
            mirostat_mode: int = 0,
            mirostat_tau: float = 5.0,
            mirostat_eta: float = 0.1,
            model: Optional[str] = None,
            logits_processor: Optional[llama.LogitsProcessorList] = None,
            grammar: Optional[llama.LlamaGrammar] = None,
            logit_bias: Optional[Dict[str, float]] = None,
            **kwargs,  # type: ignore
        ) -> Union[
            llama_types.CreateChatCompletionResponse,
            Iterator[llama_types.CreateChatCompletionStreamResponse],
        ]:
            result = f(
                messages=messages,
                functions=functions,
                function_call=function_call,
            )
            prompt = result.prompt
            if result.stop is not None:
                stop = [] if stop is None else [stop] if isinstance(stop, str) else stop
                rstop = result.stop if isinstance(result.stop, list) else [result.stop]
                stop = stop + rstop

            if response_format is not None and response_format["type"] == "json_object":
                grammar = llama_grammar.LlamaGrammar.from_string(
                    llama_grammar.JSON_GBNF
                )

            completion_or_chunks = llama.create_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                stream=stream,
                stop=stop,
                seed=seed,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_penalty=repeat_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                model=model,
                logits_processor=logits_processor,
                grammar=grammar,
                logit_bias=logit_bias,
            )
            return _convert_completion_to_chat(completion_or_chunks, stream=stream)

        register_chat_completion_handler(name)(basic_create_chat_completion)
        return f

    return decorator


def get_chat_format(name: str):
    try:
        return _CHAT_FORMATS[name]
    except KeyError:
        raise ValueError(
            f"Invalid chat format: {name} (valid formats: {list(_CHAT_FORMATS.keys())})"
        )


def hf_autotokenizer_to_chat_formatter(
    pretrained_model_name_or_path: Union[str, os.PathLike[str]]
) -> ChatFormatter:
    # https://huggingface.co/docs/transformers/main/chat_templating
    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1#instruction-format
    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/tokenizer_config.json
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

    def format_autotokenizer(
        messages: List[llama_types.ChatCompletionRequestMessage],
        **kwargs: Any,
    ) -> ChatFormatterResponse:
        tokenizer.use_default_system_prompt = False
        _prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        # Return formatted prompt and eos token by default
        return ChatFormatterResponse(prompt=_prompt, stop=tokenizer.eos_token)

    return format_autotokenizer


# see https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/tokenization_llama.py
# system prompt is "embedded" in the first message
# Assuming the existence of the `format_prompt` function with its definition,
# and other helper functions (`_get_system_message`, `_map_roles`) as needed.

# Llama-2 Format
@register_chat_format("llama_2")
def format_llama2(messages: List[llama_types.ChatCompletionRequestMessage], **kwargs: Any) -> ChatFormatterResponse:
    return format_prompt(
        messages,
        "<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>",
        "<s>[INST]", "[/INST]", " ", "</s>", "[/INST]"
    )

# Alpaca Format
@register_chat_format("alpaca")
def format_alpaca(messages: List[llama_types.ChatCompletionRequestMessage], **kwargs: Any) -> ChatFormatterResponse:
    return format_prompt(
        messages,
        "{system_message}",
        "### Instruction", "### Response", "\n\n", "</s>", ""
    )

# Miniguanaco Format
# TODO: Figure out stop
@register_chat_format("guanaco")
def format_miniguanco(messages: List[llama_types.ChatCompletionRequestMessage], **kwargs: Any) -> ChatFormatterResponse:
    return format_prompt(
        messages,
        None,
        "### Human: ", "### Assistant: ", "\n", "\n", "\n\n\n"
    )

# StableLM Format
@register_chat_format("stablelm")
def format_stablelm(messages: List[llama_types.ChatCompletionRequestMessage], **kwargs: Any) -> ChatFormatterResponse:
    # system_message = _get_system_message(messages)
    return format_prompt(
        messages,
        None,
        "<|user|>\n", "<|assistant|>\n", "<|endoftext|>\n", "<|endoftext|>\n", "<|endoftext|>\n"
    )

# Qwen Format
@register_chat_format("qwen")
def format_qwen(messages: List[llama_types.ChatCompletionRequestMessage], **kwargs: Any) -> ChatFormatterResponse:
    return format_prompt(
        messages,
        "system\n{system_message}",
        "user", "assistant", "", "", ""
    )

# Vicuna Format
@register_chat_format("vicuna")
def format_vicuna(messages: List[llama_types.ChatCompletionRequestMessage], **kwargs: Any) -> ChatFormatterResponse:
    return format_prompt(
        messages,
        "{system_message}",
        "USER", "ASSISTANT", " ", "</s>", ""
    )

# Oasst Llama Format
@register_chat_format("oasst_llama")
def format_oasst_llama(messages: List[llama_types.ChatCompletionRequestMessage], **kwargs: Any) -> ChatFormatterResponse:
    return format_prompt(
        messages,
        "[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        "", "", "</s>", "", ""
    )

# Baichuan-2 Format
@register_chat_format("baichuan_2")
def format_baichuan2(messages: List[llama_types.ChatCompletionRequestMessage], **kwargs: Any) -> ChatFormatterResponse:
    return format_prompt(
        messages,
        "{system_message}",
        "<reserved_106>", "<reserved_107>", "", "", ""
    )

# Baichuan Format
@register_chat_format("baichuan")
def format_baichuan(messages: List[llama_types.ChatCompletionRequestMessage], **kwargs: Any) -> ChatFormatterResponse:
    return format_prompt(
        messages,
        "{system_message}",
        "<reserved_102>", "<reserved_103>", "", "", ""
    )

# OpenBuddy Format
@register_chat_format("openbuddy")
def format_openbuddy(messages: List[llama_types.ChatCompletionRequestMessage], **kwargs: Any) -> ChatFormatterResponse:
    return format_prompt(
        messages,
        "{system_message}",
        "User", "Assistant", "\n", "", ""
    )

# Redpajama-Incite Format
@register_chat_format("redpajama")
def format_redpajama_incite(messages: List[llama_types.ChatCompletionRequestMessage], **kwargs: Any) -> ChatFormatterResponse:
    return format_prompt(
        messages,
        "{system_message}",
        "<human>", "<bot>", "\n", "", "<human>"
    )

# Snoozy Format
@register_chat_format("snoozy")
def format_snoozy(messages: List[llama_types.ChatCompletionRequestMessage], **kwargs: Any) -> ChatFormatterResponse:
    return format_prompt(
        messages,
        "### Instruction:\n{system_message}",
        "### Prompt", "### Response", "\n", "", "###"
    )

# Phind Format
@register_chat_format("phind")
def format_phind(messages: List[llama_types.ChatCompletionRequestMessage], **kwargs: Any) -> ChatFormatterResponse:
    return format_prompt(
        messages,
        "{system_message}",
        "### User Message", "### Assistant", "\n\n", "", ""
    )

# Intel Format
# TODO: Figure out stop
@register_chat_format("intel")
def format_intel(messages: List[llama_types.ChatCompletionRequestMessage], **kwargs: Any) -> ChatFormatterResponse:
    # system_message = _get_system_message(messages)
    return format_prompt(
        messages,
        "### System:\n{system_message}\n",
        "### User: ", "### Assistant: ", "\n", "\n", "###"
    )

# Open-Orca Format
@register_chat_format("open_orca")
def format_open_orca(messages: List[llama_types.ChatCompletionRequestMessage], **kwargs: Any) -> ChatFormatterResponse:
    return format_prompt(
        messages,
        "{system_message}",
        "User", "Assistant", "\n", "", "User"
    )

# Mistrallite Format
@register_chat_format("mistrallite")
def format_mistrallite(messages: List[llama_types.ChatCompletionRequestMessage], **kwargs: Any) -> ChatFormatterResponse:
    return format_prompt(
        messages,
        "{system_message}</s>",
        "", "</s>\n", " ", "", ""
    )

# Zephyr Format
# WORKING
@register_chat_format("zephyr")
def format_zephyr(messages: List[llama_types.ChatCompletionRequestMessage], **kwargs: Any) -> ChatFormatterResponse:
    return format_prompt(
        messages,
        "<|system|>\n{system_message}</s>\n",
        "<|user|>\n", "<|assistant|>\n", "</s>\n", "</s>\n", "</s>"
    )

# Pygmalion Format
@register_chat_format("pygmalion")
def format_pygmalion(messages: List[llama_types.ChatCompletionRequestMessage], **kwargs: Any) -> ChatFormatterResponse:
    return format_prompt(
        messages,
        "{system_message}",
        "", "", "\n", "", "\n"
    )

# ChatML Format
# WORKING
@register_chat_format("chatml")
def format_chatml(messages: List[llama_types.ChatCompletionRequestMessage], **kwargs: Any) -> ChatFormatterResponse:
    return format_prompt(
        messages,
        "<|im_start|>system\n{system_message}<|im_end|>\n",
        "<|im_start|>user\n", "<|im_start|>assistant\n", "<|im_end|>\n", "<|im_end|>\n", "<|im_end|>\n"
    )

# OpenChat Format
@register_chat_format("openchat")
def format_openchat(messages: List[llama_types.ChatCompletionRequestMessage], **kwargs: Any) -> ChatFormatterResponse:
    return format_prompt(
        messages,
        "{system_message}",
        "GPT4 Correct User: ", "GPT4 Correct Assistant: ", "", "", ""
    )

@register_chat_completion_handler("functionary")
def functionary_chat_handler(
    llama: llama.Llama,
    messages: List[llama_types.ChatCompletionRequestMessage],
    functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
    function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
    tools: Optional[List[llama_types.ChatCompletionTool]] = None,
    tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 40,
    min_p: float = 0.05,
    typical_p: float = 1.0,
    stream: bool = False,
    stop: Optional[Union[str, List[str]]] = [],
    response_format: Optional[llama_types.ChatCompletionRequestResponseFormat] = None,
    max_tokens: Optional[int] = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repeat_penalty: float = 1.1,
    tfs_z: float = 1.0,
    mirostat_mode: int = 0,
    mirostat_tau: float = 5.0,
    mirostat_eta: float = 0.1,
    model: Optional[str] = None,
    logits_processor: Optional[llama.LogitsProcessorList] = None,
    grammar: Optional[llama.LlamaGrammar] = None,
    **kwargs,  # type: ignore
) -> Union[llama_types.ChatCompletion, Iterator[llama_types.ChatCompletionChunk]]:
    SYSTEM_MESSAGE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary"""

    def generate_type_definition(
        param: Dict[str, llama_types.JsonType], indent_level: int, shared_defs
    ) -> str:
        indent = "  " * indent_level
        if "$ref" in param:
            # Reference to a shared definition
            ref_name = param["$ref"].split("/")[
                -1
            ]  # Extract the type name from the reference
            return ref_name
        elif param.get("type") == "array":
            items = param.get("items", {})
            item_type = generate_type_definition(items, indent_level + 1, shared_defs)
            return f"Array<{item_type}>"
        elif param.get("type") == "object":
            properties = param.get("properties", {})
            nested_schema = "{\n"
            for nested_param_name, nested_param in properties.items():
                nested_param_type = generate_type_definition(
                    nested_param, indent_level + 1, shared_defs
                )
                nested_schema += (
                    f"{indent}  {nested_param_name}: {nested_param_type},\n"
                )
            nested_schema += indent + "}"
            return nested_schema
        elif "enum" in param:
            # Enum type
            return " | ".join([f'"{enum_value}"' for enum_value in param["enum"]])
        else:
            # Simple type
            return param.get("type", "any")

    def generate_shared_definitions(shared_defs, indent_level: int) -> str:
        indent = "  " * indent_level
        shared_definitions = ""
        for def_name, def_properties in shared_defs.items():
            shared_definitions += f"{indent}type {def_name} = "
            if def_properties.get("type") == "object":
                shared_definitions += generate_type_definition(
                    def_properties, indent_level, shared_defs
                )
            elif "enum" in def_properties:
                # Enum type
                shared_definitions += " | ".join(
                    [f'"{enum_value}"' for enum_value in def_properties["enum"]]
                )
            shared_definitions += ";\n"
        return shared_definitions

    def generate_schema_from_functions(functions, namespace="functions") -> str:
        schema = (
            "// Supported function definitions that should be called when necessary.\n"
        )
        schema += f"namespace {namespace} {{\n\n"

        # Generate shared definitions
        shared_definitions = {}
        for function in functions:
            parameters = function.get("parameters", {})
            shared_definitions.update(parameters.get("$defs", {}))

        schema += generate_shared_definitions(shared_definitions, 1)

        for function in functions:
            function_name = function["name"]
            description = function.get("description", "")
            parameters = function.get("parameters", {})
            required_params = parameters.get("required", [])

            schema += f"  // {description}\n"
            schema += f"  type {function_name} = (_: {{\n"

            for param_name, param in parameters.get("properties", {}).items():
                param_description = param.get("description", "")
                param_type = generate_type_definition(param, 2, shared_definitions)
                optional_indicator = "" if param_name in required_params else "?"
                schema += f"    // {param_description}\n"
                schema += f"    {param_name}{optional_indicator}: {param_type},\n"
            schema += "  }) => any;\n\n"

        schema += "}} // namespace {}\n".format(namespace)
        return schema

    def prepare_messages_for_inference(
        messages: List[llama_types.ChatCompletionRequestMessage],
        functions: Optional[List[llama_types.ChatCompletionFunctions]] = None,
        tools: Optional[List[llama_types.ChatCompletionTool]] = None,
    ):
        all_messages: List[llama_types.ChatCompletionRequestMessage] = []
        if functions is not None:
            all_messages.append(
                llama_types.ChatCompletionRequestSystemMessage(
                    role="system", content=generate_schema_from_functions(functions)
                )
            )

        if tools is not None:
            all_messages.append(
                llama_types.ChatCompletionRequestSystemMessage(
                    role="system",
                    content=generate_schema_from_functions(
                        [
                            tool["function"]
                            for tool in tools
                            if tool["type"] == "function"
                        ]
                    ),
                )
            )

        all_messages.append(
            llama_types.ChatCompletionRequestSystemMessage(
                role="system", content=SYSTEM_MESSAGE
            )
        )

        for message in messages:
            # Function call responses
            if message["role"] == "function" and "name" in message:
                message["name"] = f"functions.{message['name']}"
            # Function call requests by assistant
            if "function_call" in message:
                message["function_call"][
                    "name"
                ] = f"functions.{message['function_call']['name']}"
            all_messages.append(message)

        all_messages.append(
            llama_types.ChatCompletionRequestAssistantMessage(
                role="assistant", content=None
            )
        )

        def message_to_str(msg: llama_types.ChatCompletionRequestMessage):
            if msg["role"] == "system":
                return f"system:\n{msg['content']}\n"

            elif msg["role"] == "function" and "name" in msg:
                return f"function name={msg['name']}:\n{msg['content']}\n"
            elif msg["role"] == "function" and "function_call" in msg:
                return f"function name={msg['function_call']['name']}:\n{msg['function_call']['arguments']}\n"
            elif msg["role"] == "tool":
                if msg["content"] is not None:
                    return f"function name={msg['tool_call_id']}:\n{msg['content']}\n"
                else:
                    return f"function name={msg['tool_call_id']}\n"
            elif msg["role"] == "user":
                if msg["content"] is None:
                    return "user:\n</s></s>\n"
                else:
                    return f"user:\n</s>{msg['content']}</s>\n"
            elif msg["role"] == "assistant":
                if msg["content"] is not None and "function_call" in msg:
                    return f"assistant:\n{msg['content']}\nassistant to={msg['function_call']['name']}:\n{msg['function_call']['arguments']}</s>\n"
                elif "function_call" in msg:
                    return f"assistant to={msg['function_call']['name']}:\n{msg['function_call']['arguments']}</s>\n"
                elif "tool_calls" in msg and len(msg["tool_calls"]) > 0:
                    for tool_call in msg[
                        "tool_calls"
                    ]:  # NOTE: probably doesn't work with the functionary model
                        return f"assistant to={tool_call['id']}:\n{tool_call['function']['arguments']}</s>\n"
                elif msg["content"] is None:
                    return "assistant"
                else:
                    return f"assistant:\n{msg['content']}\n"
            else:
                raise ValueError(f"Unsupported role: {msg['role']}")

        return "".join([message_to_str(msg) for msg in all_messages])

    if tools is not None:
        functions = [tool["function"] for tool in tools if tool["type"] == "function"]

    if tool_choice is not None:
        function_call = (
            tool_choice if isinstance(tool_choice, str) else tool_choice["function"]
        )

    prompt = prepare_messages_for_inference(messages, functions, tools)

    if function_call is None and (functions is None or len(functions) == 0):
        completion_or_completion_chunks = llama.create_completion(
            prompt=prompt + ":\n",
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            stream=stream,
            stop=["user:", "</s>"],
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repeat_penalty=repeat_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            logits_processor=logits_processor,
            grammar=grammar,
        )
        return _convert_completion_to_chat(completion_or_completion_chunks, stream=stream)  # type: ignore

    if function_call is None or (
        isinstance(function_call, str) and function_call == "auto"
    ):
        stop = "\n"
        completion: llama_types.Completion = llama.create_completion(
            prompt=prompt, stop=stop, stream=False
        )  # type: ignore
        completion_text = completion["choices"][0]["text"]
        # strip " to=functions." and ending ":"
        function_call = completion_text.split(".")[-1][:-1]
        new_prompt = prompt + completion_text + stop
    elif isinstance(function_call, str) and function_call != "none":
        new_prompt = prompt + f":\n"
    elif isinstance(function_call, dict):
        new_prompt = prompt + f" to=functions.{function_call['name']}:\n"
        function_call = function_call["name"]
    else:
        new_prompt = prompt + f":\n"

    function_body = None
    for function in functions or []:
        if function["name"] == function_call:
            function_body = function["parameters"]
            break
    for tool in tools or []:
        if tool["type"] == "function" and tool["function"]["name"] == function_call:
            function_body = tool["function"]["parameters"]
            break

    if function_body is not None:
        try:
            with suppress_stdout_stderr(disable=llama.verbose):
                grammar_text = llama_grammar.json_schema_to_gbnf(
                    json.dumps(function_body)
                )
                grammar = llama_grammar.LlamaGrammar.from_string(
                    llama_grammar.json_schema_to_gbnf(json.dumps(function_body))
                )
                print(grammar_text)
        except Exception as e:
            if llama.verbose:
                print(
                    "Failed to parse function body as JSON schema, falling back to default grammar"
                )
                print(e)
            with suppress_stdout_stderr(disable=llama.verbose):
                grammar = llama_grammar.LlamaGrammar.from_string(
                    llama_grammar.JSON_GBNF
                )
    else:
        with suppress_stdout_stderr(disable=llama.verbose):
            grammar = llama_grammar.LlamaGrammar.from_string(llama_grammar.JSON_GBNF)

    completion: llama_types.Completion = llama.create_completion(
        prompt=new_prompt,
        stop=["user:", "</s>"],
        stream=False,
        grammar=grammar,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        typical_p=typical_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repeat_penalty=repeat_penalty,
        tfs_z=tfs_z,
        mirostat_mode=mirostat_mode,
        mirostat_tau=mirostat_tau,
        mirostat_eta=mirostat_eta,
        model=model,
        logits_processor=logits_processor,
    )  # type: ignore

    assert "usage" in completion
    assert isinstance(function_call, str)
    assert stream is False  # TODO: support stream mode

    if llama.verbose:
        print(new_prompt)
        print(completion["choices"][0]["text"])

    # TODO: support stream mode
    return llama_types.CreateChatCompletionResponse(
        id="chat" + completion["id"],
        object="chat.completion",
        created=completion["created"],
        model=completion["model"],
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": function_call,
                        "arguments": completion["choices"][0]["text"],
                    },
                    "tool_calls": [
                        {
                            "id": function_call,
                            "type": "function",
                            "function": {
                                "name": function_call,
                                "arguments": completion["choices"][0]["text"],
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        usage=completion["usage"],
    )


class Llava15ChatHandler:
    _clip_free = None

    def __init__(self, clip_model_path: str, verbose: bool = False):
        import llama_cpp.llava_cpp as llava_cpp

        self._llava_cpp = llava_cpp
        self.clip_model_path = clip_model_path
        self.verbose = verbose
        self._clip_free = self._llava_cpp._libllava.clip_free  # type: ignore

        with suppress_stdout_stderr(disable=self.verbose):
            self.clip_ctx = self._llava_cpp.clip_model_load(
                self.clip_model_path.encode(), 0
            )

    def __del__(self):
        with suppress_stdout_stderr(disable=self.verbose):
            if self.clip_ctx is not None and self._clip_free is not None:
                self._clip_free(self.clip_ctx)
                self.clip_ctx = None

    def load_image(self, image_url: str) -> bytes:
        if image_url.startswith("data:"):
            import base64

            image_bytes = base64.b64decode(image_url.split(",")[1])
            return image_bytes
        else:
            import urllib.request

            with urllib.request.urlopen(image_url) as f:
                image_bytes = f.read()
                return image_bytes

    def __call__(
        self,
        *,
        llama: llama.Llama,
        messages: List[llama_types.ChatCompletionRequestMessage],
        functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
        function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
        tools: Optional[List[llama_types.ChatCompletionTool]] = None,
        tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        response_format: Optional[
            llama_types.ChatCompletionRequestResponseFormat
        ] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        logits_processor: Optional[llama.LogitsProcessorList] = None,
        grammar: Optional[llama.LlamaGrammar] = None,
        **kwargs,  # type: ignore
    ) -> Union[
        llama_types.CreateChatCompletionResponse,
        Iterator[llama_types.CreateChatCompletionStreamResponse],
    ]:
        assert (
            llama.context_params.logits_all is True
        )  # BUG: logits_all=True is required for llava
        assert self.clip_ctx is not None
        system_prompt = _get_system_message(messages)
        system_prompt = (
            system_prompt
            if system_prompt != ""
            else "A chat between a curious human and an artificial intelligence assistant.  The assistant gives helpful, detailed, and polite answers to the human's questions."
        )
        user_role = "\nUSER:"
        assistant_role = "\nASSISTANT:"
        llama.reset()
        llama.eval(llama.tokenize(system_prompt.encode("utf8"), add_bos=True))
        for message in messages:
            if message["role"] == "user" and message["content"] is not None:
                if isinstance(message["content"], str):
                    llama.eval(
                        llama.tokenize(
                            f"{user_role} {message['content']}".encode("utf8"),
                            add_bos=False,
                        )
                    )
                else:
                    assert isinstance(message["content"], list)
                    llama.eval(
                        llama.tokenize(f"{user_role} ".encode("utf8"), add_bos=False)
                    )
                    for content in message["content"]:
                        if content["type"] == "text":
                            llama.eval(
                                llama.tokenize(
                                    f"{content['text']}".encode("utf8"), add_bos=False
                                )
                            )
                        if content["type"] == "image_url":
                            image_bytes = (
                                self.load_image(content["image_url"]["url"])
                                if isinstance(content["image_url"], dict)
                                else self.load_image(content["image_url"])
                            )
                            import array

                            data_array = array.array("B", image_bytes)
                            c_ubyte_ptr = (
                                ctypes.c_ubyte * len(data_array)
                            ).from_buffer(data_array)
                            with suppress_stdout_stderr(disable=self.verbose):
                                embed = (
                                    self._llava_cpp.llava_image_embed_make_with_bytes(
                                        ctx_clip=self.clip_ctx,
                                        n_threads=llama.context_params.n_threads,
                                        image_bytes=c_ubyte_ptr,
                                        image_bytes_length=len(image_bytes),
                                    )
                                )
                            try:
                                n_past = ctypes.c_int(llama.n_tokens)
                                n_past_p = ctypes.pointer(n_past)
                                with suppress_stdout_stderr(disable=self.verbose):
                                    self._llava_cpp.llava_eval_image_embed(
                                        ctx_llama=llama.ctx,
                                        embed=embed,
                                        n_batch=llama.n_batch,
                                        n_past=n_past_p,
                                    )
                                assert llama.n_ctx() >= n_past.value
                                llama.n_tokens = n_past.value
                            finally:
                                with suppress_stdout_stderr(disable=self.verbose):
                                    self._llava_cpp.llava_image_embed_free(embed)
            if message["role"] == "assistant" and message["content"] is not None:
                llama.eval(
                    llama.tokenize(
                        f"ASSISTANT: {message['content']}".encode("utf8"), add_bos=False
                    )
                )
                assert llama.n_ctx() >= llama.n_tokens
        llama.eval(llama.tokenize(f"{assistant_role}".encode("utf8"), add_bos=False))
        assert llama.n_ctx() >= llama.n_tokens

        prompt = llama.input_ids[: llama.n_tokens].tolist()

        if response_format is not None and response_format["type"] == "json_object":
            with suppress_stdout_stderr(disable=self.verbose):
                grammar = llama_grammar.LlamaGrammar.from_string(
                    llama_grammar.JSON_GBNF
                )

        return _convert_completion_to_chat(
            llama.create_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                stream=stream,
                stop=stop,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_penalty=repeat_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                model=model,
                logits_processor=logits_processor,
                grammar=grammar,
            ),
            stream=stream,
        )
