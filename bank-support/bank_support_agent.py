from dataclasses import dataclass
from typing import cast

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import KnownModelName
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()


class DatabaseConn:
    """This is a fake database for example purposes.

    In reality, you'd be connecting to an external database
    (e.g. PostgreSQL) to get information about customers.
    """

    @classmethod
    async def customer_name(cls, *, id: int) -> str | None:
        customer_name_and_id = {
            1: "John",
            2: "Trump",
            3: "Praveen",
            4: "Lucky",
        }
        if id in customer_name_and_id:
            return customer_name_and_id[id]
        raise ValueError(f"Customer not found with the given id :- {id}")

    @classmethod
    async def customer_balance(cls, *, id: int, include_pending: bool) -> float:
        account_balance = {
            1: 100.0,
            2: 123.45,
            3: 342,
            4: 10,
        }
        if id in account_balance and include_pending:
            return account_balance[id]
        else:
            raise ValueError("Customer not found")


@dataclass
class SupportDependencies:
    customer_id: int
    db: DatabaseConn


class SupportResult(BaseModel):
    support_advice: str = Field(description="Advice returned to the customer")
    block_card: bool = Field(description="Whether to block their card or not")
    risk: int = Field(description="Risk level of query", ge=0, le=10)


ollama_model = OpenAIModel(
    model_name="qwen2.5:3b",  # replace with whatever model you have hosted using ollama NOTE: Make sure the llm supports tool calling
    provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
)

model = cast(KnownModelName, "groq:llama-3.3-70b-specdec")


support_agent = Agent(
    model=model,  # to use locally hosted llm replace with `ollama_model`
    deps_type=SupportDependencies,
    result_type=SupportResult,
    system_prompt=(
        "You are a support agent in our bank, give the "
        "Reply using the customer's name use `add_customer_name`."
        "customer support and judge the risk level of their query. "
    ),
    retries=2,
)


@support_agent.system_prompt
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"The customer's name is {customer_name!r}"


@support_agent.tool
async def customer_balance(
    ctx: RunContext[SupportDependencies], include_pending: bool
) -> str:
    """Returns the customer's current account balance."""
    balance = await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=True,
    )
    return f"${balance:.2f}"


if __name__ == "__main__":
    deps = SupportDependencies(customer_id=3, db=DatabaseConn())
    result = support_agent.run_sync("What is my balance?", deps=deps)
    print(result.data)
    """
    support_advice='Your account balance is $100.00. Is there anything else we can help you with?' block_card=False risk=0
    """

    result = support_agent.run_sync("I just lost my card!", deps=deps)
    print(result.data)
    """
    support_advice='We’re sorry to hear that your card is lost or stolen. Please don’t attempt to use it or your account details. For security reasons, we can block or cancel your card if you have lost it or it’s been stolen or if the information on the card is incorrect. Please call our customer service to assist you in blocking your card.' block_card=True risk=8
    """
