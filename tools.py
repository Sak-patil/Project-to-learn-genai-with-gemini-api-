""" making the current date and time tool as there are some operations which model cant do so for that 
 tools get used to do that work in user server then send that result to model again 
 so openai cant give us the exact time directly by model so for that have to write the tool 
for its result """

from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()
import os

def get_current_datetime():
    return {
        "date": datetime.now().strftime("%d-%m-%Y"),
        "time": datetime.now().strftime("%H:%M:%S")
    }
# print(get_current_datetime())

# JSON schema so that the AI knows what inputs a function requires and what type each input should be. 

get_current_datetime_schema = types.FunctionDeclaration(
    name="get_current_datetime",
    description="Returns the current date and time.",
    parameters={
        "type": "OBJECT",
        "properties": {}
    }
)

tool = types.Tool(
    function_declarations=[get_current_datetime_schema]
)

messages=[]

messages.append({
    "role":"user",
    "parts": [
        {
            "text": "What is the exact time?"
        }
    ]
})

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)

response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=messages,
        config=types.GenerateContentConfig(
          tools=[tool]  #Everything inside config tells Gemini how it is allowed to generate the response.
    ) 
)
available_tools={
    "get_current_datetime":get_current_datetime
}

tool_name=response.candidates[0].content.parts[0].function_call.name   #Extracting the name from response send by model 

result = available_tools[tool_name]() #here () has written to execute that function means from dictionary we get 
                                    #the name of tool like get_current_datetime so to execute it () is written ) 
# print(result)
# print(result["time"])
# now we got the result from our tool ,so have to send that result to model again for further process 
messages.append(
            types.Content(
                role="tool",
                parts=[
                    # It tells Gemini this message is a tool response, not ordinary text.
                    # It automatically formats the data according to Gemini's expected protocol.
                    types.Part.from_function_response(
                        name=tool_name,
                        response=result
                    )
                ]
            )
        )

final_response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=messages
)

print(final_response.text)



