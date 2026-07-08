""" in this when there is list of operation we want from gemini model and instead of making the 
separate function call for each operation using batch tool we make the only one function call 
which consist all operations to be done and then sends the result togetherly to gemini model 
instead of each result separately this reducve the number of function calls and improve performance"""

from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()
import os

def batch_calculator(expressions):

    results = []

    for expression in expressions:
        answer = eval(expression)

        results.append({
            "expression": expression,
            "result": answer
        })

    return {
        "results": results
    }


batch_calculator_schema = types.FunctionDeclaration(
    name="batch_calculator",
    description="Calculates multiple mathematical expressions.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "expressions": {
                "type": "ARRAY",
                "items": {
                    "type": "STRING"
                },
                "description": "List of mathematical expressions."
            }
        },
        "required": ["expressions"]
    }
)

tool = types.Tool(
    function_declarations=[batch_calculator_schema]
)

messages=[]
messages.append({
    "role":"user",
    "parts": [
        {
            "text": "calculate 30+30,10+10,20+20 "
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
print(response)
available_tools={
     "batch_calculator": batch_calculator,
}

function_call = response.candidates[0].content.parts[0].function_call
tool_name = function_call.name
args = dict(function_call.args)
print(tool_name)
print(args)

result = available_tools[tool_name](**args) #here () has written to execute that function means from dictionary we get 
                                    #the name of tool like get_current_datetime so to execute it () is written ) 
print(result)

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
    contents=messages,
    config=types.GenerateContentConfig(
        tools=[tool] 
    ) 
)

if __name__=="__main__":  # when we import this file tools.py this wont get print 
                          #due to this 
    print(final_response.text)
    

