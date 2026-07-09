import os

from dotenv import load_dotenv
from typing import Literal, Union

load_dotenv()
API_KEY = os.getenv("API_KEY")

from openai import OpenAI
client = OpenAI(api_key=API_KEY)

from dataclasses import dataclass
from typing import Type, TypeVar
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

Models = Literal["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]
DEFAULT_MODEL: Models = "gpt-5-mini"

@dataclass(frozen=True)
class PromptToList:
    user_prompt: str
    system_prompt: str|None = None
    message_list: list|None = None

    def __post_init__(self):
        msg_list = []
        if self.system_prompt:
            msg_list.append({"role": "system", "content": self.system_prompt})
        if self.message_list:
            msg_list.extend(self.message_list)
        msg_list.append({"role": "user", "content": self.user_prompt})
        object.__setattr__(self, "message_list", msg_list)

def list_models():
    for model in client.models.list():
        print(model.id)

def llm_basic(user_prompt:str, system_prompt:str|None = None, message_list:list|None = None, model:  Models = DEFAULT_MODEL) -> str:
    prompt = PromptToList(user_prompt=user_prompt, system_prompt=system_prompt, message_list=message_list)
    resp = client.chat.completions.create(
        model=model,
        messages=prompt.message_list)

    return resp.choices[0].message.content

def llm_structured(
    user_prompt: str,
    text_format: Type[T],
    system_prompt: str | None = None,
    message_list: list | None = None,
    model:  Models = DEFAULT_MODEL
    ) -> T | None:
    prompt = PromptToList(user_prompt=user_prompt, system_prompt=system_prompt, message_list=message_list)

    response = client.responses.parse(
        model=model, 
        input=prompt.message_list,
        text_format=text_format
    )
    
    return response.output_parsed

def format_system_prompt(prompt: str, model:  Models = DEFAULT_MODEL):
    system_prompt = """ \
        You are an Expert Prompt Engineer specializing in OpenAI's language models (gpt-5.2, etc.). Your primary objective is to take a user's draft system prompt and rewrite it to maximize clarity, performance, and reliability.

        When presented with a draft prompt, you must apply the following OpenAI prompt engineering best practices:

        1. **Assign a Clear Persona:** Ensure the prompt establishes a specific role, expertise, and tone for the AI.
        2. **Be Specific and Direct:** Remove ambiguity. Replace vague requests with concrete instructions. 
        3. **Use Delimiters:** Structure the prompt using markdown, XML tags (e.g., `<instructions>`, `<context>`), or triple quotes to clearly separate different sections, rules, and input variables.
        4. **Positive Framing:** Tell the model what it *should* do rather than what it *should not* do, whenever possible (e.g., use "Use formal language" instead of "Don't use slang").
        5. **Encourage Chain of Thought (if applicable):** If the prompt requires complex reasoning or logic, include instructions for the model to "think step-by-step" or output its reasoning before providing the final answer.
        """

    return llm_basic(user_prompt = prompt, system_prompt = system_prompt, model=model)