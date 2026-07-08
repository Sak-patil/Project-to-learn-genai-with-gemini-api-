from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()
import os

def read_file(file_name):
    with open(file_name,"r")as f:
        data=f.read()

    return{
        "content":data
    }

read_file_schema = types.FunctionDeclaration(
    name="read_file",
    description="Reads the content of a text file.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "file_name": {
                "type": "STRING",
                "description": "Name of the file to read."
            }
        },
        "required": ["file_name"]
    }
)

def write_file(file_name,content):
    with open(file_name,"w") as f:
        f.write(content)

    return{
        "status":"success"
    }

write_file_schema = types.FunctionDeclaration(
    name="write_file",
    description="Creates or overwrites a file with the given content.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "file_name": {
                "type": "STRING"
            },
            "content": {
                "type": "STRING"
            }
        },
        "required": [
            "file_name",
            "content"
        ]
    }
)

def replace(file_name,prev_word,new_word):
    with open(file_name,"r") as f:
        data = f.read()
        data = data.replace(prev_word,new_word)

    with open(file_name,"w") as f:
        f.write(data)

    return{
        "status":"sucess"
    }

replace_schema = types.FunctionDeclaration(
    name="replace_text",
    description="Replace a word or text inside a file.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "file_name": {
                "type": "STRING"
            },
            "prev_word": {
                "type": "STRING"
            },
            "new_word": {
                "type": "STRING"
            }
        },
        "required": [
            "file_name",
            "prev_word",
            "new_word"
        ]
    }
)

tool = types.Tool(
    function_declarations=[read_file_schema,write_file_schema,replace_schema]
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

available_tools={
    "read_file": read_file,
    "write_file": write_file,
    "replace_text": replace
}

while True:

    # Ask Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=messages,
        config=types.GenerateContentConfig(
            tools=[tool]
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
            tool_function = available_tools[tool_name]

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