from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        news = request.form["news"]
        data = vectorizer.transform([news])
        prediction = model.predict(data)
        result = "Fake News" if prediction[0] == 1 else "Real News"
        return render_template("index.html", prediction=result, input_text=news)

if __name__ == "__main__":
    app.run(debug=True)
