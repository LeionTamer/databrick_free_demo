import os, uuid, datetime

from dataclasses import dataclass
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Literal, List, Union, Type, TypeVar
from openai import OpenAI
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, IntegerType

from shared.catalog_manager import CatalogManager

token_schema = StructType([
    StructField("id", StringType(), True),
    StructField("timestamp", TimestampType(), True),
    StructField("model_name", StringType(), True),
    StructField("input_tokens", IntegerType(), True),
    StructField("output_tokens", IntegerType(), True)
])

token_table = CatalogManager(fqn=f"clinton_emails.bronze.token_usage", schema=token_schema)

load_dotenv()
API_KEY = os.getenv("API_KEY")
T = TypeVar('T', bound=BaseModel)

client = OpenAI(api_key=API_KEY)

Models = Literal["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]
DEFAULT_MODEL: Models = "gpt-5-mini"

Embedding_Models = Literal["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"

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

def llm_basic(
    user_prompt:str,
    system_prompt:str|None = None,
    message_list:list|None = None,
    model:  Models = DEFAULT_MODEL
    ) -> str:
    prompt = PromptToList(user_prompt=user_prompt, system_prompt=system_prompt, message_list=message_list)
    resp = client.chat.completions.create(
        model=model,
        messages=prompt.message_list)
    
    usage = resp.usage

    token_table.add_entry({
        "id":str(uuid.uuid4()),
        "timestamp":datetime.datetime.now(),
        "model_name":model,
        "input_tokens":usage.prompt_tokens,
        "output_tokens":usage.completion_tokens})

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
        You are an Expert Prompt Engineer specializing in OpenAI's language models (GPT-4, GPT-4o, etc.). Your primary objective is to take a user's draft system prompt and rewrite it to maximize clarity, performance, and reliability.

        When presented with a draft prompt, you must apply the following OpenAI prompt engineering best practices:

        1. **Assign a Clear Persona:** Ensure the prompt establishes a specific role, expertise, and tone for the AI.
        2. **Be Specific and Direct:** Remove ambiguity. Replace vague requests with concrete instructions. 
        3. **Use Delimiters:** Structure the prompt using markdown, XML tags (e.g., `<instructions>`, `<context>`), or triple quotes to clearly separate different sections, rules, and input variables.
        4. **Positive Framing:** Tell the model what it *should* do rather than what it *should not* do, whenever possible (e.g., use "Use formal language" instead of "Don't use slang").
        5. **Specify the Output Format:** Clearly define how the final output should look (e.g., JSON, markdown, structured headers, bulleted lists).
        6. **Encourage Chain of Thought (if applicable):** If the prompt requires complex reasoning or logic, include instructions for the model to "think step-by-step" or output its reasoning before providing the final answer.

        **YOUR REQUIRED OUTPUT FORMAT**
        Whenever the user provides a draft prompt, you must respond with the following strict structure:

        ### 1. Critique & Improvements
        Provide a brief, bulleted summary (2-4 points) explaining the weaknesses in the original prompt and how you improved them based on best practices.

        ### 2. Refined System Prompt
        Provide the newly optimized system prompt inside a markdown code block. Ensure it is ready to be copied and pasted directly into an API or ChatGPT system prompt field. Use placeholders like `{{VARIABLE_NAME}}` if the prompt requires dynamic user input at runtime.

        ### 3. Usage Notes (Optional)
        Briefly explain how the user should provide their user-side input to get the best results with this new system prompt.
        """

    return llm_basic(user_prompt = prompt, system_prompt = system_prompt, model=model)

def get_embedding(text: str, model: Embedding_Models = DEFAULT_EMBEDDING_MODEL) -> List[float]:
    return client.embeddings.create(
        model=model,
        input=text
    ).data[0].embedding

def commit_tokens():
    token_table.commit()