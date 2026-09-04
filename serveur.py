import os
import random
from pathlib import Path

import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

RACINE = Path(__file__).resolve().parent
DOSSIER_MODELE_SEQ2SEQ = RACINE / "modeles"
TEMPERATURE_PAR_DEFAUT = 0.0
LONGUEUR_MAX_ENTREE = 2000

app = Flask(__name__)
CORS(app)  # Autorise les requêtes venant d'autres origines (comme Flutter Web)

model = joblib.load(RACINE / "modele_intention.pkl")
vectorizer = joblib.load(RACINE / "vectorizer_intention.pkl")
reponses = joblib.load(RACINE / "reponses_intention.pkl")

# Le seq2seq n'existe qu'une fois `entrainer.py` lancé : le serveur doit démarrer sans.
repondeur = None
erreur_seq2seq = None
try:
    from fatia.generation import Repondeur

    repondeur = Repondeur(DOSSIER_MODELE_SEQ2SEQ)
except Exception as exception:
    erreur_seq2seq = f"{type(exception).__name__}: {exception}"


def lire_texte():
    """Valide l'entrée. Renvoie (texte, réponse_erreur) — l'un des deux est None."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None, (jsonify({"erreur": "Corps JSON attendu"}), 400)

    texte = data.get("texte")
    if not isinstance(texte, str) or not texte.strip():
        return None, (jsonify({"erreur": "Aucun texte fourni"}), 400)

    if len(texte) > LONGUEUR_MAX_ENTREE:
        return None, (
            jsonify({"erreur": f"Texte trop long (max {LONGUEUR_MAX_ENTREE} caractères)"}),
            400,
        )

    return texte.strip(), None


@app.route("/sante", methods=["GET"])
def sante():
    return jsonify({
        "classifieur": True,
        "intentions": sorted(reponses),
        "seq2seq": repondeur is not None,
        "seq2seq_epoch": repondeur.epoch if repondeur else None,
        "seq2seq_perte_validation": repondeur.perte_validation if repondeur else None,
        "seq2seq_erreur": erreur_seq2seq,
    })


@app.route("/predire", methods=["POST"])
def predire():
    texte, erreur = lire_texte()
    if erreur:
        return erreur

    X_phrase = vectorizer.transform([texte])
    intention = model.predict(X_phrase)[0]
    reponse = random.choice(reponses[intention])

    return jsonify({
        "texte_recu": texte,
        "intention": intention,
        "reponse": reponse,
    })


@app.route("/discuter", methods=["POST"])
def discuter():
    if repondeur is None:
        return jsonify({
            "erreur": "Modèle seq2seq non chargé. Lance d'abord entrainer.py.",
            "detail": erreur_seq2seq,
        }), 503

    texte, erreur = lire_texte()
    if erreur:
        return erreur

    donnees = request.get_json(silent=True) or {}
    temperature = donnees.get("temperature", TEMPERATURE_PAR_DEFAUT)
    if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 5:
        return jsonify({"erreur": "temperature doit être un nombre entre 0 et 5"}), 400

    resultat = repondeur.repondre(texte, temperature=float(temperature))

    return jsonify({
        "texte_recu": texte,
        "reponse": resultat["reponse"],
        "repli": resultat["repli"],
        "tokens_inconnus": resultat["tokens_inconnus"],
        "tokens_total": resultat["tokens_total"],
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
