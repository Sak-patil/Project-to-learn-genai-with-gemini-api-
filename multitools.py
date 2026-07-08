from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()
import os
import tools 
from datetime import datetime
"""here as we have imported the file tools.py to use the lists and functions 
presnent in that file but importing prints all the print(),variables,statement from that file
which are outside the functions omporting only defines the function not 
execute it but in our tools.py file there are statements like response=client.  ...
due to which it is sending the query to gemini model so it is not good practice
so when importing remember that import that file which has functions and not statements like 
client.model.generate .... """
"""so here we have """
def get_current_day():
    return {
        "day": datetime.now().strftime("%A")
    }

get_current_day_schema = types.FunctionDeclaration(
    name="get_current_day",
    description="Returns the current day of the week.",
    parameters={
        "type": "OBJECT",
        "properties": {}
    }
)

messages = []
print("empty messages")
input_msg=input(">")

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)

# Add first user message
messages.append(
    types.Content(
        role="user",
        parts=[
            types.Part(text=input_msg)
        ]
    )
)


while True:

    # Ask Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=messages,
        config=types.GenerateContentConfig(
            tools=[tools.tool]
        )
    )

    # Save Gemini's complete response
    model_message = response.candidates[0].content
    messages.append(model_message)

    tool_called = False

    # Check every part of Gemini's response
    for part in model_message.parts:

        if part.function_call:

            tool_called = True

            tool_name = part.function_call.name
            args = dict(part.function_call.args)

            print(f"Calling Tool : {tool_name}")
            print(args)

            # -------------------------
            # Execute the correct tool
            # -------------------------
            tool_function = tools.available_tools[tool_name]

            if tool_function is None:
                raise Exception(f"Unknown Tool: {tool_name}")

            result = tool_function(**args) #for arguments
            
            # -------------------------
            # Send tool response back
            # -------------------------

            messages.append(
                types.Content(
                    role="tool",
                    parts=[
                        types.Part.from_function_response(
                            name=tool_name,
                            response=result
                        )
                    ]
                )
            )

    # No tool requested -> Final answer
    if not tool_called:
        print(response.text)
        break
