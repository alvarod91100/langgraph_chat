from graph import app
from pprint import pprint

user_question = None
while user_question != "exit":
    user_question = input("Enter a question: ")
    inputs = {"question": user_question}
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")
    pprint(value["generation"])