import os
from typing import cast

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName

load_dotenv()


class MyModel(BaseModel):
    city: str
    state: str
    country: str
    population: str


model = cast(
    KnownModelName,
    os.getenv(
        "PYDANTIC_AI_MODEL", "groq:llama-3.1-8b-instant"
    ),  # set GROQ_API_KEY first
)
print(f"Using model: {model}")
agent = Agent(model, result_type=MyModel, instrument=True)

if __name__ == "__main__":
    result = agent.run_sync("The windy city in the US of A.")
    print(result.data)
    print(result.usage())
