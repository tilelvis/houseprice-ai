import functions_framework
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

@functions_framework.http
def predict(request):
    data = request.get_json()
    features = np.array([[
        data["MedInc"], data["HouseAge"], data["AveRooms"],
        data["AveBedrms"], data["Population"],
        data["AveOccup"], data["Latitude"], data["Longitude"]
    ]])
    prediction = model.predict(features)[0]
    return {"EstimatedPrice": float(prediction*100000)}