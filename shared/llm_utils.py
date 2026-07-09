from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("API_KEY")

from openai import OpenAI
client = OpenAI(api_key=API_KEY)

from dataclasses import dataclass
from typing import Type, TypeVar
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

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

def llm_basic(user_prompt:str, system_prompt:str|None = None, message_list:list|None = None) -> str:
    prompt = PromptToList(user_prompt=user_prompt, system_prompt=system_prompt, message_list=message_list)
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=prompt.message_list)

    return resp.choices[0].message.content

def llm_structured(
    user_prompt: str,
    text_format: Type[T],
    system_prompt: str | None = None,
    message_list: list | None = None
    ) -> T | None:
    prompt = PromptToList(user_prompt=user_prompt, system_prompt=system_prompt, message_list=message_list)

    response = client.responses.parse(
        model="gpt-4o", 
        input=prompt.message_list,
        text_format=text_format
    )
    
    return response.output_parsed