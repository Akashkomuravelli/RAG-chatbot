"""LangChain retrieval-augmented generation pipeline."""

from __future__ import annotations

import json
from urllib.parse import quote
from urllib.request import urlopen
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from .config import Settings

SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions using the provided retrieved context.

Use the retrieved context as the primary source of truth for knowledge-base questions. Do not invent facts that are not supported by it.
If the answer cannot be determined from the context, clearly state that the information was not found in the provided knowledge base.
For current weather questions, use the get_weather tool instead of guessing. Include the location in your answer.
Be concise, accurate, and helpful.

Retrieved context:
{context}
"""


@tool
def get_weather(location: str) -> str:
    """Get the current weather and today's forecast for a named location."""
    geocode_url = (
        "https://geocoding-api.open-meteo.com/v1/search?name="
        f"{quote(location)}&count=1&language=en&format=json"
    )
    with urlopen(geocode_url, timeout=10) as response:
        geocode_data = json.load(response)
    places = geocode_data.get("results", [])
    if not places:
        return f"I could not find a location named {location}."

    place = places[0]
    latitude = place["latitude"]
    longitude = place["longitude"]
    forecast_url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}"
        "&current=temperature_2m,relative_humidity_2m,apparent_temperature,"
        "weather_code,wind_speed_10m&daily=temperature_2m_max,temperature_2m_min,"
        "precipitation_probability_max&timezone=auto"
    )
    with urlopen(forecast_url, timeout=10) as response:
        forecast_data = json.load(response)

    current = forecast_data["current"]
    current_units = forecast_data["current_units"]
    daily = forecast_data["daily"]
    daily_units = forecast_data["daily_units"]
    place_name = ", ".join(
        value for value in (place.get("name"), place.get("country")) if value
    )
    return (
        f"Weather for {place_name}: {current['temperature_2m']}"
        f"{current_units['temperature_2m']}, feels like {current['apparent_temperature']}"
        f"{current_units['apparent_temperature']}, humidity {current['relative_humidity_2m']}%"
        f", wind {current['wind_speed_10m']} {current_units['wind_speed_10m']}. "
        f"Today's high is {daily['temperature_2m_max'][0]}"
        f"{daily_units['temperature_2m_max']} and low is {daily['temperature_2m_min'][0]}"
        f"{daily_units['temperature_2m_min']}; precipitation probability is "
        f"{daily['precipitation_probability_max'][0]}%."
    )


@dataclass
class RAGResponse:
    answer: str
    sources: list[Document]


class RAGPipeline:
    """Retrieve relevant chunks and pass them to a Groq chat model."""

    def __init__(self, settings: Settings, vector_store) -> None:
        self.settings = settings
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.retrieval_k},
        )
        self.llm = ChatGroq(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            temperature=0,
            max_retries=2,
        )
        self.llm_with_tools = self.llm.bind_tools([get_weather])
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", "Question: {question}")]
        )

    def ask(self, question: str) -> RAGResponse:
        """Retrieve context, generate a grounded answer, and return source chunks."""
        question = question.strip()
        if not question:
            raise ValueError("Please enter a question.")

        documents = self.retriever.invoke(question)
        context = "\n\n---\n\n".join(document.page_content for document in documents)
        messages = self.prompt.invoke(
            {"context": context or "No relevant context was retrieved.", "question": question}
        ).to_messages()
        response = self.llm_with_tools.invoke(messages)

        if response.tool_calls:
            messages.append(response)
            for tool_call in response.tool_calls:
                if tool_call["name"] == "get_weather":
                    messages.append(get_weather.invoke(tool_call))
            response = self.llm.invoke(messages)

        answer = response.content if isinstance(response.content, str) else str(response.content)
        return RAGResponse(answer=answer, sources=documents)
