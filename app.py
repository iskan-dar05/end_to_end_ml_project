from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np


with open("models.pkl", "rb") as f:
    loaded_models = pickle.load(f)

    

app = Flask(__name__)


@app.route("/health")
    