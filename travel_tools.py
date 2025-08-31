from agents import function_tool

@function_tool
def get_flight(destination: str) -> str:
    return f"Flights found to {destination}: PKR 45,000 - PKR70,000."

@function_tool
def suggest_hotels(destination: str) -> str:
    return f"Hotels in {destination}: Pearl Continental, Marriot, Local Guest Houses."

