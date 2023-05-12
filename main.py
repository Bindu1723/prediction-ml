import pickle
from flask import Flask, render_template, request
from scipy.special import inv_boxcox
from os.path import join, dirname

filename = join(dirname(__file__), "model", "final_model.pkl")
app = Flask(__name__)
lambda_ = -0.34318951188220137


@app.route("/")
def hello_word():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def find_price():
    model = pickle.load(open(filename, "rb"))
    bhk = int(request.form["bhk"])
    size = int(request.form["size"])
    area_type = int(request.form["areaType"])
    city = int(request.form["city"])
    furnishing_status = int(request.form["furnishingStatus"])
    tenant_preferred = int(request.form["tenantsPreferred"])
    bathroom = int(request.form["bathroom"])
    point_of_contact = int(request.form["pointOfContact"])
    floor_level = int(request.form["floorLevel"])
    total_floors = int(request.form["totalFloors"])
    month_posted = int(request.form["month"])
    day_posted = int(request.form["day"])
    day_of_the_week_posted = int(request.form["week"])
    quarter_posted = int(request.form["quarter"])

    x = [
        [
            bhk,
            size,
            area_type,
            city,
            furnishing_status,
            tenant_preferred,
            bathroom,
            point_of_contact,
            floor_level,
            total_floors,
            month_posted,
            day_posted,
            day_of_the_week_posted,
            quarter_posted,
        ]
    ]

    prediction = model.predict(x)
    rent = inv_boxcox(prediction, lambda_)[0]
    rent = round(rent, -1)
    return render_template("result.html", rent=rent)


if __name__ == "__main__":
    app.run(debug=True)
