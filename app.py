from flask import Flask,render_template,request
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np


app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

vehicle_class = {'COMPACT': 0.0,
 'SUV - SMALL': 11.0,
 'MID-SIZE': 2.0,
 'TWO-SEATER': 13.0,
 'MINICOMPACT': 3.0,
 'SUBCOMPACT': 10.0,
 'FULL-SIZE': 1.0,
 'STATION WAGON - SMALL': 9.0,
 'SUV - STANDARD': 12.0,
 'VAN - CARGO': 14.0,
 'VAN - PASSENGER': 15.0,
 'PICKUP TRUCK - STANDARD': 6.0,
 'MINIVAN': 4.0,
 'SPECIAL PURPOSE VEHICLE': 7.0,
 'STATION WAGON - MID-SIZE': 8.0,
 'PICKUP TRUCK - SMALL': 5.0}

fuel_type = {'Z': 4.0, 'D': 0.0, 'X': 3.0, 'E': 1.0, 'N': 2.0}

transmission = {'AS5': 14.0,
 'M6': 25.0,
 'AV7': 22.0,
 'AS6': 15.0,
 'AM6': 8.0,
 'A6': 3.0,
 'AM7': 9.0,
 'AV8': 23.0,
 'AS8': 17.0,
 'A7': 4.0,
 'A8': 5.0,
 'M7': 26.0,
 'A4': 1.0,
 'M5': 24.0,
 'AV': 19.0,
 'A5': 2.0,
 'AS7': 16.0,
 'A9': 6.0,
 'AS9': 18.0,
 'AV6': 21.0,
 'AS4': 13.0,
 'AM5': 7.0,
 'AM8': 10.0,
 'AM9': 11.0,
 'AS10': 12.0,
 'A10': 0.0,
 'AV10': 20.0}

@app.route("/")
def home():
    return render_template('index.html',fuel_type=fuel_type,vehicle_class=vehicle_class,
            transmission=transmission)

sc = StandardScaler()
@app.route("/predict",methods=["POST"])
def predict():
    features = [x for x in request.form.items()]
    features = dict(features)
    order_of_features = ["vehicle_class","transmission","fuel_type","engine_size","cylinders","fuel_consumption_c"]
    X = []
    for i in order_of_features:
        X.append(features[i])
    X = np.array(X).reshape(-1,1).T # row : 1 ; col : 6 (1,6)
    return render_template('index.html',fuel_type=fuel_type,vehicle_class=vehicle_class,
            transmission=transmission,predicted_value=model.predict(X)[0])
    

if __name__ == "__main__":
    app.run()
