import flask
from flask import request
import json
import time

from model import generate_model

app = flask.Flask(__name__, static_url_path="", static_folder="public")

@app.post("/generate_model")
def handle_generate_model():
    prompt = request.json["prompt"]

    print(f"recieved a generation prompt \"{prompt}\"")

    # TODO: query our trained model with the prompt
    start_time = time.time()
    model = generate_model(prompt)
    end_time = time.time()

    return flask.Response(
        json.dumps({
            "generationTime": f"{end_time - start_time}s",
            "model": model,
        }),
        status=200,
        headers={
            "Content-Type": "text/javascript; charset=utf-8",
        },
    )

app.run(host="0.0.0.0", port="20010", debug=True)