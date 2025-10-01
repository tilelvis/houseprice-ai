import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

def train_and_save_model():
    """
    Fetches the California housing dataset, trains a Linear Regression model,
    and saves it to model.pkl.
    """
    # Load the dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    # Create and train the model
    model = LinearRegression()
    model.fit(X, y)

    # Save the model directly into the functions directory
    joblib.dump(model, "functions/model.pkl")
    print("Model trained and saved as functions/model.pkl")

if __name__ == "__main__":
    train_and_save_model()