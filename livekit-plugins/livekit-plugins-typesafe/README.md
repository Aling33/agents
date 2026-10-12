# TypeSafe plugin for LiveKit Agents

Support for [TypeSafe](https://typesafe.ai) Jev decision models. Jev evaluates probability
questions, named choices, and ordered scores over a conversation in one request.

## Installation

```bash
pip install livekit-plugins-typesafe
```

## Pre-requisites

Set `TYPESAFE_API_KEY` for the TypeSafe API, or `OPENROUTER_API_KEY` to use Jev through
OpenRouter's Decisions API.

## Background decisions

```python
from livekit.agents import Agent, AgentSession, decisions
from livekit.plugins import typesafe

session = AgentSession(
    decision_model=typesafe.DecisionModel(model="jev-1.13.0"),
    # or: typesafe.DecisionModel.with_openrouter()
    ...,
)

agent = Agent(
    instructions="You are a restaurant receptionist.",
    decisions={
        "callback": decisions.Probability("The caller wants a staff member to call back."),
        "intent": decisions.Choice(
            "What does the caller want?",
            options={"booking": "Make a reservation", "other": "Anything else"},
        ),
        "frustration": decisions.Score(
            "How frustrated is the caller?", levels=["Calm", "Annoyed", "Angry"]
        ),
    },
)


@session.on("decisions_completed")
def on_decisions(ev: decisions.DecisionsCompletedEvent) -> None:
    for name, result in ev.results.items():
        print(name, result.value)
```

## On-demand evaluation

```python
from livekit.agents import decisions, llm, utils
from livekit.plugins import typesafe

async with utils.http_context.open():
    model = typesafe.DecisionModel()
    context = llm.ChatContext.empty()
    context.add_message(role="user", content="Please let me speak to a person.")
    response = await model.evaluate(
        chat_ctx=context,
        decisions={"handoff": decisions.Probability("The caller requests a human.")},
    )
```

Each requested decision appears in either `response.results` or `response.errors`.
Request failures and malformed responses raise.

## Notes

- Jev rounds probabilities and scores to two decimal places. Validation allows for this
  rounding, and results keep the provider's values.
- Score answers must return a legend that matches the requested levels.
- `provider_data["confidence"]` describes how concentrated the distribution is, not the
  probability that the answer is correct.
- Jev accepts text only. System and developer messages, agent instructions, and non-text
  content are never sent.
