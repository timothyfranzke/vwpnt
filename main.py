from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from fastapi.middleware.cors import CORSMiddleware
from langchain.output_parsers import PydanticOutputParser
from datetime import datetime
from geopy.distance import geodesic
import os
import json

app = FastAPI()

# Enable CORS for all domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
os.environ["OPENAI_API_KEY"] = "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"
# Ensure you have set the OPENAI_API_KEY environment variable
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize the ChatOpenAI model
chat_model = ChatOpenAI()

class GeoPoint(BaseModel):
    latitude: float
    longitude: float

class FirestoreTimestamp(BaseModel):
    seconds: int
    nanoseconds: int

class TimeSpan(BaseModel):
    start: FirestoreTimestamp
    end: FirestoreTimestamp

    @validator('start', 'end', pre=True)
    def parse_timestamp(cls, v):
        if isinstance(v, dict):
            return FirestoreTimestamp(**v)
        return v

    def to_datetime(self, timestamp: FirestoreTimestamp) -> datetime:
        return datetime.utcfromtimestamp(timestamp.seconds + timestamp.nanoseconds / 1e9)

class Event(BaseModel):
    id: str
    name: str
    slug: str
    geoLocation: GeoPoint
    timeSpan: TimeSpan

class SearchRequest(BaseModel):
    prompt: str
    events: List[Event]

def create_event_context(events: List[Event]) -> str:
    context = "Here is a list of available events:\n\n"
    for event in events:
        context += f"- Name: {event.name}\n"
        context += f"  Slug: {event.slug}\n"
        context += f"  Location: {event.geoLocation.latitude}, {event.geoLocation.longitude}\n"
        context += f"  Start: {event.timeSpan.to_datetime(event.timeSpan.start).isoformat()}\n"
        context += f"  End: {event.timeSpan.to_datetime(event.timeSpan.end).isoformat()}\n\n"
    return context

@app.get("/") 
async def root():
    return "Hello World"

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/search_events")
async def search_events(request: SearchRequest):
    try:
        # Create context from the provided events
        event_context = create_event_context(request.events)

        # Define the prompt template with event context
        prompt_template = ChatPromptTemplate.from_template(
            "You are an AI assistant specialized in searching for events. "
            "You have access to the following events:\n\n"
            "{event_context}\n"
            "Given the following search prompt, analyze the available events and return a list of event slugs "
            "that best match the search criteria. If no events match, return an empty list.\n"
            "Search prompt: {prompt}\n"
            "Provide the response as a JSON array of event slugs."
        )

        # Generate the matching event slugs
        chain = prompt_template | chat_model
        result = chain.invoke({
            "event_context": event_context,
            "prompt": request.prompt
        })

        # Parse the JSON response
        try:
            matching_slugs = json.loads(result.content)
        except json.JSONDecodeError:
            raise ValueError("Failed to parse AI response as JSON")

        # Filter events based on the matching slugs
        matching_events = [
            event for event in request.events
            if event.slug in matching_slugs
        ]

        # Convert matching events to dict for JSON serialization
        matching_events_dict = [event.dict() for event in matching_events]

        return matching_events_dict

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)