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
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=history,
        #giving system instructions 
        config=GenerateContentConfig(
            temperature=temperature,
            system_instruction="You are a math tutor do not directly give ans give hint only "
        )
    )

    return response.text

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

    # print(history)
    #builted the simple chat bot in which the request(user question) and response
    #giveby model is saved in history (list) so that model can remember the 
    #previous context as this increases the token usability cause we are 
    #history also thats why modern models dont use this method 

    #whats happening 
    #1.user's question gets saved in histpry list 
    #2.history list send to model
    #3.the respomse of model also getting saved in history list
    #4.this happens on loop (while:True)