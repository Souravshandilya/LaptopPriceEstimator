import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import pickle

# Load the pickle file
with open('updated_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Check the model's type and library references
print(f"Model type: {type(model)}")
print(f"Model modules: {model.__module__}")

# Update the module references (if necessary)
def update_library_reference(obj):
    # Recursively update module references
    if hasattr(obj, '__module__') and obj.__module__.startswith('sklearn'):
        obj.__module__ = obj.__module__.replace('sklearn', 'scikit-learn')
    if hasattr(obj, '__dict__'):
        for key, value in obj.__dict__.items():
            update_library_reference(value)

update_library_reference(model)

# Save the updated model
with open('updated_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("updated_linear_model.pickle saved to 'updated_model.pkl'")


# Initialize Flask app
app = Flask(__name__)

# Load the trained model
try:
    model_path = 'updated_model.pkl'  # Correct path to the model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
    print("Model type:", type(model))
    print("Attributes:", dir(model))
except Exception as e:
    print("Error loading model:", e)
    model = None

# Load the dataset for dropdown options
try:
    data = pd.read_csv('laptops_train.csv')
    data.columns = data.columns.str.strip()
    print("Dataset loaded successfully.")
except Exception as e:
    print("Error loading dataset:", e)
    data = None

# Extract unique values for dropdown options
def get_dropdown_options(column_name):
    try:
        if data is not None:
            return sorted(data[column_name].dropna().unique().tolist())
        else:
            raise ValueError("Dataset is not loaded.")
    except Exception as e:
        print(f"Error extracting options for {column_name}:", e)
        return []

# Create dropdown options
dropdown_options = {
    "Screen Size": get_dropdown_options("Screen Size"),
    "CPU": get_dropdown_options("CPU"),
    "RAM": get_dropdown_options("RAM"),
    "Storage": get_dropdown_options("Storage"),
    "GPU": get_dropdown_options("GPU"),
    "Weight": get_dropdown_options("Weight"),
}
print("Dropdown options initialized:", dropdown_options)

@app.route('/')
def home():
    print("Rendering home page.")
    return render_template('index.html', dropdown_options=dropdown_options)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Error: Model not loaded. Please check the model file.", 500

    try:
        # Debug: Log form data
        print("Form Data Received:", request.form)
        print("Predict endpoint accessed.")

        # Encode features
        encoded_features = []
        for feature, value in request.form.items():
            print(f"Processing feature '{feature}' with value '{value}'")
            if feature not in dropdown_options:
                raise ValueError(f"Feature '{feature}' not in dropdown options.")
            if value not in dropdown_options[feature]:
                print(f"Available options for '{feature}': {dropdown_options[feature]}")
                raise ValueError(f"Value '{value}' for feature '{feature}' not in dropdown options.")
            encoded_features.append(dropdown_options[feature].index(value))

        # Debug: Log encoded features
        print("Encoded Features:", encoded_features)

        # Check input dimensions
        features_array = np.array([encoded_features])
        print("Features array for prediction:", features_array)

        # Prediction
        prediction = model.predict(features_array)[0]
        print("Prediction Result:", prediction)

        # Render result
        return render_template(
            'index.html',
            dropdown_options=dropdown_options,
            prediction_text=f'Predicted Price: {round(prediction, 2)}'
        )
    except Exception as e:
        print("Error Occurred:", e)
        return f"Error: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
