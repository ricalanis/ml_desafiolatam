from flask import Flask
import pickle
from flask import request
import numpy as np

rfrgs = pickle.load(open("model_14122017.pickle", "rb"))

app = Flask(__name__)

@app.route("/")
def hello():
    rm = float(request.args.get("rm"))
    lstat = float(request.args.get("lstat"))
    y_predicted = rfrgs.predict([rm, lstat])
    return str(y_predicted)


if __name__ == '__main__':
    app.run(debug=True)