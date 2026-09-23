"""Output parsers: turn raw model text into structured Python values.

An output parser does two things: (1) `get_format_instructions()` — text you
splice into the prompt telling the model exactly how to format its reply,
and (2) `.parse()` / `.invoke()` — code that turns the model's string output
back into a Python object (list, dict, or a validated Pydantic model).
"""

from langchain_core.output_parsers import (
    CommaSeparatedListOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from core import get_model


class MovieReview(BaseModel):
    title: str = Field(description="the movie's title")
    rating: int = Field(description="rating out of 10")
    one_line_verdict: str = Field(description="a single-sentence verdict")


def main() -> None:
    model = get_model()

    # 1. StrOutputParser: the default, just unwraps `.content` from AIMessage.
    str_parser = StrOutputParser()
    chain = ChatPromptTemplate.from_template("Say hello to {name} in French.") | model | str_parser
    print("--- StrOutputParser ---")
    print(repr(chain.invoke({"name": "Kris"})))

    # 2. CommaSeparatedListOutputParser: model output -> Python list[str].
    list_parser = CommaSeparatedListOutputParser()
    list_prompt = ChatPromptTemplate.from_template(
        "List 5 {topic}.\n{format_instructions}"
    ).partial(format_instructions=list_parser.get_format_instructions())
    chain = list_prompt | model | list_parser
    result = chain.invoke({"topic": "prime numbers under 30"})
    print("\n--- CommaSeparatedListOutputParser ---")
    print(result, type(result))

    # 3. JsonOutputParser: freeform JSON -> dict, no schema enforced.
    json_parser = JsonOutputParser()
    json_prompt = ChatPromptTemplate.from_template(
        "Return a JSON object with keys 'city' and 'country' for: {place}\n"
        "{format_instructions}"
    ).partial(format_instructions=json_parser.get_format_instructions())
    chain = json_prompt | model | json_parser
    result = chain.invoke({"place": "the city with the Eiffel Tower"})
    print("\n--- JsonOutputParser ---")
    print(result, type(result))

    # 4. PydanticOutputParser: JSON -> validated, typed Python object.
    pydantic_parser = PydanticOutputParser(pydantic_object=MovieReview)
    review_prompt = ChatPromptTemplate.from_template(
        "Write a short review of {movie}.\n{format_instructions}"
    ).partial(format_instructions=pydantic_parser.get_format_instructions())
    chain = review_prompt | model | pydantic_parser
    result = chain.invoke({"movie": "The Matrix"})
    print("\n--- PydanticOutputParser ---")
    print(result, type(result))
    print("Rating field:", result.rating)


if __name__ == "__main__":
    main()
