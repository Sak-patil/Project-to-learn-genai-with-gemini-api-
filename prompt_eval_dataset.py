import os
from google import genai
from dotenv import load_dotenv
load_dotenv()
from google.genai.types import GenerateContentConfig
import json

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)

prompt="""Generate an evaluation dataset for a prompt evaluation.

The dataset will be used to evaluate prompts that generate Python, JSON, or Regex solutions for programming tasks.

Generate an array of JSON objects, where each object represents one programming task.

Example output:

```json
[
    {
        "task": "Description of task"
    },
    {
        "task": "Description of another task"
    }
]
```

Requirements:

* Focus on tasks that can be solved by writing a single Python function, a single JSON object, or a single regular expression.
* Focus on tasks that do not require writing much code.
* Keep each task short and clear (1–2 sentences).
make 3 objects only 
Return ONLY valid JSON.
Do NOT include markdown code fences.
Do NOT include explanations or any extra text. """

def generate_dataset(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text

dataset=generate_dataset(prompt)
print(dataset)
with open("evaluation_dataset.json", "w") as file:
    json.dump(json.loads(dataset), file, indent=4)