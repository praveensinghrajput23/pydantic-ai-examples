import asyncio
import os
from dataclasses import dataclass
from typing import Any, cast

import logfire
from devtools import debug
from dotenv import load_dotenv
from httpx import AsyncClient
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import KnownModelName

load_dotenv()

logfire.configure(
    send_to_logfire=True,
    service_name=os.getenv("LOGFIRE_SERVICE_NAME"),
    service_version=os.getenv("LOGFIRE_SERVICE_VERSION"),
    token=os.getenv("LOGFIRE_TOKEN"),
    environment=os.getenv("LOGFIRE_ENVIRONMENT"),
)


@dataclass
class WeatherAgentDep:
    """WeatherAgentDep is a dataclass that represents a weather agent dependency."""

    client: AsyncClient
    weather_api_key: str | None
    geo_api_key: str | None


weather_agent = Agent(
    model=cast(KnownModelName, "groq:llama-3.1-8b-instant"),
    name="weather_agent",
    deps_type=WeatherAgentDep,
    retries=2,
    instrument=True,
)


@weather_agent.tool
async def get_lat_lng(
    ctx: RunContext[WeatherAgentDep], location_description: str
) -> dict[str, float]:
    """Get latitude and longitude from a location description."""

    if ctx.deps.geo_api_key is None:
        raise ValueError("Geo API key is required")
        # return {"lat": 51.1, "lng": -0.2} ## one can use a dummy value if there is no Weather API Key.

    params = {
        "q": location_description,
        "api_key": ctx.deps.geo_api_key,
    }
    with logfire.span("calling geocode API", params=params) as span:
        r = await ctx.deps.client.get("https://geocode.maps.co/search", params=params)
        r.raise_for_status()
        data = r.json()
        span.set_attribute("response", data)

    if data:
        return {"lat": data[0]["lat"], "lng": data[0]["lon"]}
    else:
        raise ModelRetry("Could not find the location")


@weather_agent.tool
async def get_weather(
    ctx: RunContext[WeatherAgentDep], lat: float, lng: float
) -> dict[str, Any]:
    """Get weather data from a latitude and longitude."""
    if ctx.deps.weather_api_key is None:
        raise ValueError("Weather API key is required")
        # return {"temperature": "21 °C", "description": "Sunny"}  ## one can use a dummy value if there is no Weather API Key.

    params = {
        "apikey": ctx.deps.weather_api_key,
        "location": f"{lat},{lng}",
        "units": "metric",
    }
    with logfire.span("calling weather API", params=params) as span:
        r = await ctx.deps.client.get(
            "https://api.tomorrow.io/v4/weather/realtime", params=params
        )
        r.raise_for_status()
        data = r.json()
        span.set_attribute("response", data)

    values = data["data"]["values"]

    # https://docs.tomorrow.io/reference/data-layers-weather-codes
    code_lookup = {
        1000: "Clear, Sunny",
        1100: "Mostly Clear",
        1101: "Partly Cloudy",
        1102: "Mostly Cloudy",
        1001: "Cloudy",
        2000: "Fog",
        2100: "Light Fog",
        4000: "Drizzle",
        4001: "Rain",
        4200: "Light Rain",
        4201: "Heavy Rain",
        5000: "Snow",
        5001: "Flurries",
        5100: "Light Snow",
        5101: "Heavy Snow",
        6000: "Freezing Drizzle",
        6001: "Freezing Rain",
        6200: "Light Freezing Rain",
        6201: "Heavy Freezing Rain",
        7000: "Ice Pellets",
        7101: "Heavy Ice Pellets",
        7102: "Light Ice Pellets",
        8000: "Thunderstorm",
    }
    return {
        "temperature": f"{values['temperatureApparent']:0.0f}°C",
        "description": code_lookup.get(values["weatherCode"], "Unknown"),
    }


async def main():
    async with AsyncClient() as client:
        # create a free API key at https://www.tomorrow.io/weather-api/
        weather_api_key = os.getenv("WEATHER_API_KEY")
        # create a free API key at https://geocode.maps.co/
        geo_api_key = os.getenv("GEO_API_KEY")
        deps = WeatherAgentDep(
            client=client, weather_api_key=weather_api_key, geo_api_key=geo_api_key
        )
        result = await weather_agent.run(
            "What is the weather like in Delhi?",
            deps=deps,
        )
        debug(result)
        print("Response:", result.data)


if __name__ == "__main__":
    asyncio.run(main())
