import os
from google import genai
from dotenv import load_dotenv
load_dotenv()
from google.genai.types import GenerateContentConfig

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)

def add_user_message(history, text):
    user_message = {
        "role": "user",
        "parts": [
            {
                "text": text
            }
        ]
    }

    history.append(user_message)

def add_model_message(history, text):
    model_message = {
        "role": "model",
        "parts": [
            {
                "text": text
            }
        ]
    }

    history.append(model_message)

def chat(history,temperature):
    full_response=""
    for chunk in client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=history
    ):
        print(chunk.text, end="", flush=True)
        full_response += chunk.text

    print()  # Move to the next line after streaming is complete
    return full_response

#making empty list for history
history = []
while True:
    print("Ask question:")
    question=input(">")
    add_user_message(history,question)

    #pass the list of history to gemini 
    answer=chat(history,1)# here 1 is temperature 
    print(answer)

    #now add that response again to the history 
    add_model_message(history,answer)