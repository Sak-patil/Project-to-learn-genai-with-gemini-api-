import os
from google import genai
from dotenv import load_dotenv
load_dotenv()

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

def chat(history):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=history
    )

    return response.text

#making empty list for history
history = []

add_user_message(history,"define quantum mechanics in one sentence")

#pass the list of history to gemini 
answer=chat(history)
print(answer)

#now add that response again to the history 
add_model_message(history,answer)

print(history)