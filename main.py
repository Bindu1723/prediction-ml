import pickle
from flask import Flask, render_template, request
from scipy.special import inv_boxcox
from os.path import join, dirname
import numpy as np

filename = join(dirname(__file__), "model", "predictor.pkl")
model = pickle.load(open(filename, "rb"))


app = Flask(__name__)


@app.route("/")
def hello_word():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def find_price():
    raw_features = [int(x) for x in request.form.values()]
    features = [np.array(raw_features)]

    prediction = model.predict(features)
    rent = round(prediction[0])
    return render_template("result.html", rent=rent)


if __name__ == "__main__":
    app.run(debug=True)
