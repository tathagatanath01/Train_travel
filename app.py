
import streamlit as st
import sqlite3
import requests
import uuid

from typing import TypedDict, Annotated

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

# LangGraph imports
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.graph.state import START, END, StateGraph

# Tavily import
from langchain_tavily import TavilySearch


# =====================================================
# 1. STREAMLIT CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Travel Assistant",
    page_icon="🚆",
    layout="wide"
)


# =====================================================
# 2. API KEYS
# =====================================================

# Replace these placeholders with your newly generated API keys.
GROQ_API_KEY = "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"
weather_api_key = "b65988ddae944bf993993731262105"
IRCTC_API_KEY = "re_live_27cb751a2d64cba6d8f75f664eb4ef66"
TAVILY_API_KEY = "tvly-4eT6lE-NVyOg8k50jhsmE0Z4yqr0HEIPlQJsMkfZi9D3l5YqX"


# =====================================================
# 3. GROQ MODEL
# =====================================================

model = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=GROQ_API_KEY
)


# =====================================================
# 4. SQLITE CHECKPOINTER
# =====================================================

@st.cache_resource
def get_checkpointer():

    conn = sqlite3.connect(
        "my_db.db",
        check_same_thread=False
    )

    checkpointer = SqliteSaver(conn)

    checkpointer.setup()

    return checkpointer


checkpointer = get_checkpointer()


# =====================================================
# 5. WEATHER TOOL
# =====================================================

@tool
def weather_tool(to_city: str):
    """Generate weather information for a city."""

    url = "https://api.weatherapi.com/v1/current.json"

    params = {
        "key": weather_api_key,
        "q": to_city
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        return {
            "City": data["location"]["name"],
            "Region": data["location"]["region"],
            "Country": data["location"]["country"],
            "Temperature": data["current"]["temp_c"],
            "Condition": data["current"]["condition"]["text"],
            "Condition_icon": data["current"]["condition"]["icon"],
            "Humidity": data["current"]["humidity"]
        }

    except requests.exceptions.RequestException as e:

        return {
            "error": f"Weather API error: {str(e)}"
        }

    except (KeyError, ValueError) as e:

        return {
            "error": f"Invalid weather response: {str(e)}"
        }


# =====================================================
# 6. TRAIN SEARCH TOOL
# =====================================================

@tool
def train_tool(from_city: str, to_city: str):
    """
    Search for trains between two stations.

    Args:
        from_city: Source station code or city.
        to_city: Destination station code or city.
    """

    url = "https://api.irctc.dev/api/v1/trains/search"

    params = {
        "from": from_city,
        "to": to_city
    }

    headers = {
        "x-api-key": IRCTC_API_KEY,
        "accept": "application/json"
    }

    try:

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=20
        )

        response.raise_for_status()

        result = response.json()

        train_data = result.get("data", [])

        trains = []

        for train in train_data:

            trains.append({

                "train_number": train.get("trainNo"),
                "train_name": train.get("trainName"),
                "train_type": train.get("trainType"),

                "from_station": train.get("fromStationName"),
                "from_code": train.get("fromStation"),

                "to_station": train.get("toStationName"),
                "to_code": train.get("toStation"),

                "departure": train.get("departure"),
                "arrival": train.get("arrival"),

                "duration": train.get("durationStr"),
                "duration_minutes": train.get("durationMin"),

                "running_days": train.get("runningDays", []),
                "classes": train.get("classes", []),

                "distance_km": train.get("distance"),
                "estimated_fare": train.get("estimatedFare"),

                "halts": train.get("halts")

            })

        return {
            "source": from_city,
            "destination": to_city,
            "total_trains": len(trains),
            "trains": trains
        }

    except requests.exceptions.RequestException as e:

        return {
            "source": from_city,
            "destination": to_city,
            "error": f"Train API error: {str(e)}",
            "total_trains": 0,
            "trains": []
        }


# =====================================================
# 7. TAVILY SEARCH
# =====================================================

search = TavilySearch(
    tavily_api_key=TAVILY_API_KEY,
    max_results=5,
    topic="general"
)


# =====================================================
# 8. HOTEL SEARCH TOOL
# =====================================================

@tool
def hotel_tool(to_city: str):
    """
    Find hotels in a given city.
    """

    prompt = f"""
    Find hotels in {to_city}, India.

    Provide:

    1. Hotel names
    2. Hotel addresses or locations
    3. Contact phone numbers, if available
    4. Official website, if available
    5. Approximate price range, if available

    Return relevant and useful hotel information.
    Do not invent contact numbers or prices.
    """

    try:

        response = search.invoke({
            "query": prompt
        })

        return {
            "hotel_information": response
        }

    except Exception as e:

        return {
            "error": f"Hotel search error: {str(e)}"
        }


# =====================================================
# 9. CITY INFORMATION TOOL
# =====================================================

@tool
def city_tool(to_city: str):
    """
    Give details about culture, history, people,
    food and tourist places of a given city.
    """

    prompt = f"""
    Give me a detailed guide about {to_city}.

    Cover these topics:

    1. Culture
    2. History
    3. People and languages
    4. Famous food and local cuisine
    5. Tourist places and attractions

    Use clear headings and bullet points.
    """

    try:

        response = search.invoke({
            "query": prompt
        })

        return {
            "city_information": response
        }

    except Exception as e:

        return {
            "error": f"City search error: {str(e)}"
        }


# =====================================================
# 10. LANGGRAPH STATE
# =====================================================

class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

    from_city: str
    to_city: str

    train_result: dict
    weather_result: dict
    hotel_result: dict
    city_result: dict

    final_result: str


# =====================================================
# 11. TRAIN NODE
# =====================================================

def train_node(state: ChatState):

    from_city = state["from_city"]
    to_city = state["to_city"]

    response = train_tool.invoke({
        "from_city": from_city,
        "to_city": to_city
    })

    return {
        "train_result": response
    }


# =====================================================
# 12. HOTEL NODE
# =====================================================

def hotel_node(state: ChatState):

    to_city = state["to_city"]

    response = hotel_tool.invoke({
        "to_city": to_city
    })

    return {
        "hotel_result": response
    }


# =====================================================
# 13. CITY NODE
# =====================================================

def city_node(state: ChatState):

    to_city = state["to_city"]

    response = city_tool.invoke({
        "to_city": to_city
    })

    return {
        "city_result": response
    }


# =====================================================
# 14. WEATHER NODE
# =====================================================

def weather_node(state: ChatState):

    to_city = state["to_city"]

    response = weather_tool.invoke({
        "to_city": to_city
    })

    return {
        "weather_result": response
    }


# =====================================================
# 15. FINAL NODE
# =====================================================

def final_tool(state: ChatState):

    prompt = f"""
    You are a professional travel guide.

    Generate detailed train information:

    {state['train_result']}

    Generate detailed hotel information:

    {state['hotel_result']}

    City Information:

    {state['city_result']}

    Weather Information:

    {state['weather_result']}

    Prepare a comprehensive travel guide.

    Include:

    1. Train details
    2. Departure and arrival time
    3. Train duration and fare, if available
    4. Hotel details
    5. Current weather
    6. Culture
    7. History
    8. People and languages
    9. Famous food
    10. Tourist destinations

    Use clear headings and bullet points.

    Do not invent train schedules, fares or hotel contact details.
    Mention when information is unavailable.
    """

    response = model.invoke(prompt)

    return {
        "final_result": response.content,
        "messages": [
            AIMessage(content=response.content)
        ]
    }


# =====================================================
# 16. LANGGRAPH WORKFLOW
# =====================================================

graph = StateGraph(ChatState)

graph.add_node("train_node", train_node)
graph.add_node("hotel_node", hotel_node)
graph.add_node("city_node", city_node)
graph.add_node("weather_node", weather_node)
graph.add_node("final_tool", final_tool)

graph.add_edge(START, "train_node")
graph.add_edge("train_node", "hotel_node")
graph.add_edge("hotel_node", "city_node")
graph.add_edge("city_node", "weather_node")
graph.add_edge("weather_node", "final_tool")
graph.add_edge("final_tool", END)

app = graph.compile(
    checkpointer=checkpointer
)


# =====================================================
# 17. STREAMLIT USER INTERFACE
# =====================================================

st.title("🚆 AI Travel Assistant")

st.subheader(
    "Your Personal AI-Powered Travel Planner"
)

st.write(
    "Plan your journey with AI. "
    "Search trains, hotels, weather and tourist attractions."
)

st.divider()


# =====================================================
# 18. USER INPUT
# =====================================================

st.header("Plan Your Journey")

col1, col2 = st.columns(2)

with col1:

    from_city = st.text_input(
        "Departure Station Code",
        placeholder="Example: MLDT"
    )

with col2:

    to_city = st.text_input(
        "Destination Station Code",
        placeholder="Example: HWH"
    )


# =====================================================
# 19. SESSION MANAGEMENT
# =====================================================

if "thread_id" not in st.session_state:

    st.session_state.thread_id = str(uuid.uuid4())


if "travel_result" not in st.session_state:

    st.session_state.travel_result = None


# =====================================================
# 20. GENERATE TRAVEL GUIDE
# =====================================================

if st.button(
    "🚀 Generate Travel Guide",
    type="primary",
    use_container_width=True
):

    if not from_city.strip() or not to_city.strip():

        st.warning(
            "Please enter both departure and destination station codes."
        )

    else:

        with st.spinner(
            "AI is preparing your travel guide. Please wait..."
        ):

            try:

                result = app.invoke(

                    {
                        "messages": [
                            HumanMessage(
                                content=f"I want to go from {from_city} to {to_city}"
                            )
                        ],

                        "from_city": from_city.strip().upper(),
                        "to_city": to_city.strip().upper()
                    },

                    config={
                        "configurable": {
                            "thread_id": st.session_state.thread_id
                        }
                    }
                )

                st.session_state.travel_result = result

                st.success(
                    "Your travel guide has been generated successfully!"
                )

            except Exception as e:

                st.error(
                    f"Something went wrong: {str(e)}"
                )


# =====================================================
# 21. DISPLAY RESULTS
# =====================================================

if st.session_state.travel_result is not None:

    result = st.session_state.travel_result

    st.divider()

    st.header("✈️ Your Travel Guide")

    st.markdown(
        result.get("final_result", "No travel guide generated.")
    )

    st.divider()

    st.header("📋 Detailed Travel Information")


    # Train information

    with st.expander("🚆 Train Information", expanded=True):

        train_result = result.get("train_result", {})

        st.json(train_result)


    # Hotel information

    with st.expander("🏨 Hotel Information"):

        hotel_result = result.get("hotel_result", {})

        st.json(hotel_result)


    # Weather information

    with st.expander("🌤️ Weather Information"):

        weather_result = result.get("weather_result", {})

        if "error" not in weather_result:

            st.metric(
                "Temperature",
                f"{weather_result.get('Temperature', 'N/A')} °C"
            )

            st.write(
                "**City:**",
                weather_result.get("City", "N/A")
            )

            st.write(
                "**Condition:**",
                weather_result.get("Condition", "N/A")
            )

            st.write(
                "**Humidity:**",
                f"{weather_result.get('Humidity', 'N/A')}%"
            )

        else:

            st.error(weather_result["error"])


    # City information

    with st.expander("🏙️ City Information"):

        city_result = result.get("city_result", {})

        st.json(city_result)


    st.divider()

    if st.button("🔄 Plan Another Journey"):

        st.session_state.travel_result = None

        st.session_state.thread_id = str(uuid.uuid4())

        st.rerun()


# =====================================================
# 22. FOOTER
# =====================================================

st.divider()

st.caption(
    "AI Travel Assistant | Powered by LangGraph, Groq, "
    "Tavily, WeatherAPI and IRCTC API @ Tathagata Nath"
)
