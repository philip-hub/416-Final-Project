import flask
from flask import request
import json

from model import generate_model

app = flask.Flask(__name__, static_url_path="", static_folder="public")

@app.post("/generate_model")
def handle_generate_model():
    prompt = request.json["prompt"]

    print(f"recieved a generation prompt \"{prompt}\"")

    # TODO: query our trained model with the prompt
    model = generate_model(prompt)

    return flask.Response(
        json.dumps({ "model": model }),
        status=200,
        headers={
            "Content-Type": "text/javascript; charset=utf-8",
        },
    )

app.run(host="0.0.0.0", port="20010", debug=True)