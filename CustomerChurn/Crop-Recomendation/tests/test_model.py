import pickle
import os
import pytest
import numpy as np

# Load the model once for all tests
@pytest.fixture(scope='module')
def model():
    assert os.path.exists("RF.pkl"), "Model file not found!"
    with open("RF.pkl", "rb") as f:
        return pickle.load(f)

def test_model_loaded(model):
    assert model is not None

def test_model_prediction_shape(model):
    # Example input: [N, P, K, temperature, humidity, pH, rainfall]
    sample = np.array([[90, 40, 40, 20.5, 80.0, 6.5, 200]])
    pred = model.predict(sample)
    assert len(pred) == 1, "Prediction should return one result"

def test_model_predicts_known_output(model):
    # This is hypothetical. Replace with real expected if known
    sample = np.array([[90, 40, 40, 20.5, 80.0, 6.5, 200]])
    pred = model.predict(sample)[0]
    assert pred in model.classes_, f"Prediction {pred} not in model classes"

def test_invalid_input_shape(model):
    with pytest.raises(ValueError):
        # Invalid input (too few features)
        model.predict(np.array([[90, 40, 40]]))

def test_prediction_type(model):
    sample = np.array([[100, 50, 60, 25.0, 70.0, 6.0, 150]])
    pred = model.predict(sample)
    assert isinstance(pred[0], str), "Prediction should return a crop name string"
