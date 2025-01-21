import os
import dotenv
from openai import OpenAI

from literature.constants import SYSTEM_PROMPT_RESEARCH_ASSISTANT

dotenv.load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def gpt_chat_completion(prompt:str, system_prompt: str = SYSTEM_PROMPT_RESEARCH_ASSISTANT) -> str | None:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o",
    )
    return chat_completion.choices[0].message.content
