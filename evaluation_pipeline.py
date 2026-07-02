import os
import json
from google import genai
from google.genai.types import GenerateContentConfig
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)

temperature = 0.2

# Run Prompt
#this is for one task of dataset
def run_prompt(test_case):
    prompt = f"""
Please solve the following task.

Task:
{test_case["task"]}

Return only the required output.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=GenerateContentConfig(
            temperature=temperature
        )
    )

    return response.text

#Grading 
def grade(test_case, output):

    grading_prompt = f"""
    You are an AI evaluator.

    Task:
    {test_case["task"]}

    Model Output:
    {output}

    Evaluate whether the model output correctly solves the task.

    Give a score from 0 to 10.

    Return ONLY a JSON object in this format:

    {{
        "score": 8,
        "reason": "Short explanation"
    }}
    give code in json format only dont give any extra explanation of symbols like'''
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=grading_prompt,
        config=GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json" #this is gemini special argument to use it as json
                                                #as theoutput is coming in ''' ''' so thats why previously code throwing error 
                                                #hence have to write this to get output in json 
        )
    )

    return json.loads(response.text)

def run_test_case(test_case):

    output = run_prompt(test_case)

    grading = grade(test_case, output)

    return {
        "test_case": test_case,
        "output": output,
        "score": grading["score"],
        "reason": grading["reason"]
    }

#now for each task in the dataset have to do the evaluation 
# Run Evaluation
def run_eval(dataset):

    results = []

    for test_case in dataset:
        result = run_test_case(test_case)
        results.append(result)

    return results

# Load Dataset
with open("evaluation_dataset.json", "r") as file:
    dataset = json.load(file)

# Run Evaluation
results = run_eval(dataset)

# Print Results
print(json.dumps(results, indent=4))