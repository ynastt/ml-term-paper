from flask import Flask, request
from flask_cors import CORS
import pandas as pd
import pickle
# import joblib
# MinMaxScaler - это класс, который используется для масштабирования признаков на определенный диапазон
from sklearn.preprocessing import MinMaxScaler
# Для геокодирования и получения широты и долготы населенного пункта или города
from geopy.geocoders import Nominatim


# Настройка Flask
app = Flask(__name__)
# Добавляем поддержку CORS (Cross-Origin Resource Sharing) для обработки AJAX-запросов
CORS(app)


# Настройка данных
# Из файла jupyter notebook analysis.ipynb
# Загружаем обученную модель для дальнейшего использования
model = pickle.load(open('model.pkl', 'rb'))
# model = joblib.load('model1.pkl')

# Из файла jupyter notebook analysis.ipynb
tornado_df = pd.read_csv('cleaned.csv')


# Маршруты Flask
# Получение пользовательского ввода из формы и прогноз категории торнадо
@app.route("/predict", methods=["POST"])
def predict():
    # Преобразование пользовательского ввода в числовые значения и сохранение в переменные
    if request.method == "POST":
        leng = float(request.form["leng"])
        wid = float(request.form["wid"])
        fat = float(request.form["fat"])
        place = request.form["place"]
        print(f'Place: {place}, len: {leng}, wid: {wid}, fatality: {fat}')

        # Преобразование наименования места в широту и долготу
        locator = Nominatim(user_agent="myAppGeocoder")
        print(f'Locator: {locator}')
        # print(locator.geocode("florida"))
        # print(locator.geocode("Texas"))
        location = locator.geocode(place)
        print(f'Location: {location}')
        slat = location.latitude
        slon = location.longitude
        print("Latitude", slat)
        print("Longitude", slon)

        # Сохранение пользовательского ввода в виде датафрейма user_df
        data = [[fat, leng, wid, slat, slon]]
        user_df = pd.DataFrame(data, columns=["fat", "len", "wid", "slat", "slon"])

        # Сохранение датафрейма, используемого для модели, как features
        features = tornado_df

        # Добавление данных пользователя к данным о торнадо
        # complete = features.append(user_df)
        complete = pd.concat([features, user_df]).reset_index(drop=True)
        
        # Масштабируем данные, т.к масштабирование признаков может улучшить процесс обучения и повысить точность модели.
        # Создается экземпляр scaler для подготовки Min-Max масштабирования данных.
        scaler = MinMaxScaler()
        # Метод fit_transform сначала вычисляет минимальное и максимальное значение каждого признака в наборе 
        # данных, а затем масштабирует признаки, применяя следующую формулу:

        # X_(scaled) = (X - X_(min))/(X_(max) - X_(min))
        # где   X_(min) - минимальное значение признака, 
        #       X_(max) - максимальное значение признака, 
        # а     X - исходное значение признака.
        scaled_df = scaler.fit_transform(complete)

        # Предсказывание категории торнадо
        # output = model.predict([list(scaled_df[len(scaled_df)-1])])
        output = model.predict([list(scaled_df[-1])])
        print(f'output ef: {output[0]}')

        # Rename output to user friendly text
        category = ""
        if(output[0] == 0):
            category = "EF 0 - Light damage"  # Легкое повреждение
        elif(output[0] == 1):
            category = "EF 1 - Moderate damage" # Умеренное повреждение
        elif(output[0] == 2):
            category = "EF 2 - Considerable damage" # Значительное повреждение
        elif(output[0] == 3):
            category = "EF 3 - Severe damage" # Тяжелые повреждения
        elif(output[0] == 4):
            category = "EF 4 - Devastating damage" # Разрушительные повреждения
        elif(output[0] == 5):
            category = "EF 5 - Incredible damage" # Невероятные повреждения
        # category = "test"
        print(category)
        return { "classify": category }

# Запуск приложения
if __name__ == "__main__":
    app.run(debug=False)
