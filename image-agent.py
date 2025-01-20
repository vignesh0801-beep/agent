from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize agent with Groq
agent = Agent(
    model=Groq(
        id="mixtral-8x7b-32768",  # Mixtral model through Groq
    ),
    tools=[DuckDuckGo()],
    instructions="Always provide detailed analysis and include sources for information",
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
)

def analyze_image_and_search(image_url: str, query: str):
    """
    Analyze image and search for additional context using Groq and DuckDuckGo.

    Args:
    image_url (str): URL of the image to analyze.
    query (str): User's query about the image.
    """
    try:
        # Construct messages in the correct format
        messages = [
            {"role": "system", "content": "You are an AI agent. Please analyze images and provide detailed responses."},
            {"role": "user", "content": f"Analyze the image at this URL: {image_url} and answer the query: {query}"}
        ]

        # Run the agent with the formatted messages
        response = agent.run(messages=messages, stream=True)
        for chunk in response:
            print(chunk)

    except Exception as e:
        print(f"Error processing image: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Test with an image
    image_url = "https://en.wikipedia.org/wiki/Kalinga_Institute_of_Industrial_Technology#/media/File:Kiit_library_building.jpg"
    query = "Tell me about the location and purpose of the building. Include any recent news or developments."

    # Analyze image and get response
    analyze_image_and_search(image_url, query)