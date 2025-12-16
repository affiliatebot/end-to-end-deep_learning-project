
from flask import Flask, request, jsonify, render_template_string
from cnnClassifier.pipeline.prediction import Prediction

app = Flask(__name__)

predictor = Prediction()

@app.route("/", methods=["GET"])
def home():
    with open("templates/index.html") as f:
        return render_template_string(f.read())
    
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]          # FileStorage object
    image_bytes = file.read()              # bytes
    result = predictor.predict(image_bytes)
    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(debug=True)
