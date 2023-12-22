from flask import Flask, request
from flask_cors import CORS
import pandas as pd

# Imports for predictions
import pickle
# import joblib
from sklearn.preprocessing import MinMaxScaler

# For geocoding to get lat long from a city/town
from geopy.geocoders import Nominatim

#################################################
# Flask Setup


app = Flask(__name__)
CORS(app)

#################################################
# Data Setup

# From jupyter notebook file
model = pickle.load(open('model.pkl', 'rb'))
# Загрузка модели для использования
# model = joblib.load('model1.pkl')

# From jupyter notebook file
tornado_df = pd.read_csv('cleaned.csv')

#################################################
# Flask Routes

# Get user input from form and predict tornado category
@app.route("/predict", methods=["POST"])
def predict():

    # convert user input from template.html to float and save as variables
    if request.method == "POST":
        leng = float(request.form["leng"])
        wid = float(request.form["wid"])
        fat = float(request.form["fat"])
        place = request.form["place"]

        print(f'place: {place}, len: {leng}, wid: {wid}, fatality: {fat}')

        # convert place to lat and long
        # иногда вылетает geocoder, поэтому можно поменять название агента, чтобы заработало
        locator = Nominatim(user_agent="myAppGeocoder")
        print(f'locator: {locator}')
        # print(locator.geocode("florida"))
        # print(locator.geocode("Texas"))
        print(locator.geocode(place))
        location = locator.geocode(place)
        print(f'location: {location}')
        slat = location.latitude
        slon = location.longitude

        print("Latitude", slat)
        print("Longitude", slon)

        # store user inputs as dataframe user_df
        data = [[fat, leng, wid, slat, slon]]
        user_df = pd.DataFrame(data, columns=["fat", "len", "wid", "slat", "slon"])
        # store dataframe used for model as features
        features = tornado_df
        # append user data to tornado data
        # complete = features.append(user_df)
        complete = pd.concat([features, user_df]).reset_index(drop=True)
        # set up scaler and apply scaling
        scaler = MinMaxScaler()
        scaled_df = scaler.fit_transform(complete)
        # Category prediction of user input from model
        # output = model.predict([list(scaled_df[len(scaled_df)-1])])
        output = model.predict([list(scaled_df[-1])])
        print(f'output ef: {output[0]}')
        # Rename output to user friendly text
        category = ""
        if(output[0] == 0):
            category = "EF 0 - Light damage"
        elif(output[0] == 1):
            category = "EF 1 - Moderate damage"
        elif(output[0] == 2):
            category = "EF 2 - Considerable damage"
        elif(output[0] == 3):
            category = "EF 3 - Severe damage"
        elif(output[0] == 4):
            category = "EF 4 - Devastating damage"
        elif(output[0] == 5):
            category = "EF 5 - Incredible damage"
        # category = "test EF div"
        print(category)
        
        return { "classify": category }


# To run applicaton
if __name__ == "__main__":
    app.run(debug=False)
