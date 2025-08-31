import os
from dotenv import load_dotenv
from agents import Agent , Runner ,AsyncOpenAI ,OpenAIChatCompletionsModel
from agents.run import RunConfig
from travel_tools import get_flight, suggest_hotels


load_dotenv()

# Client Setup for Connecting to Gemini
external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

#Initialize model
model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)
# Configuration for the Run
config = RunConfig(
    model=model,
    tracing_disabled=True
)

destination_agent = Agent(
    name= "DestinationAgent",
    instructions="you recommend travel destinations based on user preferences.",
    model=model
)
booking_agent = Agent(
    name= "BookingAgent",
    instructions= "you give flight and hotel booking options for the chosen destination.",
    model=model,
    tools=[get_flight, suggest_hotels]
)
explore_agent = Agent(
    name="ExploreAgent",
    instructions="You suggest food & places to explore in the destination.",
    model=model
)

def main():
    print("\U0001F30D AI Travel Designer\n")
    
    while True:
        mood = input("✈️ What's your travel mood (relaxing/adventure/etc)? → ")

        result1 = Runner.run_sync(destination_agent, mood, run_config=config)
        dest = result1.final_output.strip()
        print("\n📍 Destination Suggested:", dest)

        result2 = Runner.run_sync(booking_agent, dest, run_config=config)
        print("\n✈️ Booking Info:", result2.final_output)

        result3 = Runner.run_sync(explore_agent, dest, run_config=config)
        print("\n🍽️ Explore Tips:", result3.final_output)

        again = input("\n🧳 Want to plan another trip? (yes/no): ").lower()
        if again not in ('yes', 'y'):
            print("\n ✨😊 Thanks for using AI Travel Designer!")
            break

if __name__ == "__main__":
    main()









