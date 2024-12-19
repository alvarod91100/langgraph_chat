from graph import app
from pprint import pprint

inputs = {"question": "What mechanism is used to implement attention in a neural network? Explain in detail."}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
pprint(value["generation"])