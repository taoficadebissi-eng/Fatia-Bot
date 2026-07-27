from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import random

app = Flask(__name__)
CORS(app)  # Autorise les requêtes venant d'autres origines (comme Flutter Web)

model = joblib.load("modele_intention.pkl")
vectorizer = joblib.load("vectorizer_intention.pkl")
reponses = joblib.load("reponses_intention.pkl")

@app.route("/predire", methods=["POST"])
def predire():
    data = request.get_json()
    phrase = data.get("texte", "")

    if not phrase:
        return jsonify({"erreur": "Aucun texte fourni"}), 400

    X_phrase = vectorizer.transform([phrase])
    intention = model.predict(X_phrase)[0]
    reponse = random.choice(reponses[intention])

    return jsonify({
        "texte_recu": phrase,
        "intention": intention,
        "reponse": reponse
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)