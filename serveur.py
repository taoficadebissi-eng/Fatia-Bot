import os
import random
from pathlib import Path

import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai
from google.genai import types

RACINE = Path(__file__).resolve().parent
TEMPERATURE_PAR_DEFAUT = 0.7
LONGUEUR_MAX_ENTREE = 2000
# Modèle Gemini utilisé par /discuter. On peut le changer sans toucher au code
# avec la variable Render GEMINI_MODEL. « gemini-2.5-flash » n'est plus
# proposé aux nouveaux comptes (erreur 404 NOT_FOUND) : Google recommande
# « gemini-3.8-flash ».
MODELE_GEMINI = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
LLM_API_KEY = os.environ.get("LLM_API_KEY")

SEUIL_CONFIANCE = 0.22
MESSAGES_INCOMPRIS = [
    "Je n'ai pas bien compris. Tu peux reformuler ?",
    "Désolé, je n'ai pas saisi. Tu peux le dire autrement ?",
    "Là je sèche ! Reformule et je réessaie.",
]

app = Flask(__name__)
CORS(app)

model = joblib.load(RACINE / "modele_intention.pkl")
vectorizer = joblib.load(RACINE / "vectorizer_intention.pkl")
reponses = joblib.load(RACINE / "reponses_intention.pkl")

# Délai maximal d'un appel à Gemini (en millisecondes). Sans limite, un appel
# lent (quota dépassé, Google surchargé) dépassait les 30 s de gunicorn : le
# serveur était coupé net et renvoyait une page « Internal Server Error »
# (500) au lieu d'une erreur propre. Avec 20 s, /discuter répond toujours une
# erreur JSON à temps, et l'app Nolia bascule sur /predire.
DELAI_GEMINI_MS = int(os.environ.get("GEMINI_TIMEOUT_MS", "20000"))

# Niveau de « réflexion » de Gemini avant de répondre (voir /discuter).
# gemini-3.8-flash refuse MINIMAL (400 INVALID_ARGUMENT) : LOW est le plus bas.
NIVEAU_REFLEXION = os.environ.get("GEMINI_THINKING", "low").upper()


def options_http_gemini():
    """Délai maximal + une seule tentative (pas de nouvelles tentatives
    silencieuses qui allongent l'attente). `HttpRetryOptions` n'existe que dans
    les versions récentes de google-genai : sinon, on garde seulement le délai."""
    try:
        return types.HttpOptions(
            timeout=DELAI_GEMINI_MS,
            retry_options=types.HttpRetryOptions(attempts=1),
        )
    except (AttributeError, TypeError, ValueError):
        return types.HttpOptions(timeout=DELAI_GEMINI_MS)


# Le client Gemini est créé uniquement si la clé est présente dans l'environnement.
gemini_client = (
    genai.Client(api_key=LLM_API_KEY, http_options=options_http_gemini())
    if LLM_API_KEY
    else None
)


# Mémoire de la conversation : l'app peut envoyer les derniers échanges avec
# chaque question (champ « historique »), pour que Fatia comprenne la suite
# (« et à Porto-Novo ? » après une question sur Cotonou).
HISTORIQUE_MAX_MESSAGES = 10


def lire_historique(donnees):
    """Transforme le champ « historique » reçu en messages pour Gemini.

    Format attendu : [{"role": "user" | "bot", "texte": "..."}, ...], du plus
    ancien au plus récent. Tout élément mal formé est ignoré ; on ne garde que
    les HISTORIQUE_MAX_MESSAGES derniers, chacun coupé à LONGUEUR_MAX_ENTREE."""
    brut = donnees.get("historique")
    if not isinstance(brut, list):
        return []
    messages = []
    for element in brut[-HISTORIQUE_MAX_MESSAGES:]:
        if not isinstance(element, dict):
            continue
        texte = element.get("texte")
        role = element.get("role")
        if not isinstance(texte, str) or not texte.strip() or role not in ("user", "bot"):
            continue
        messages.append(
            types.Content(
                # Pour Gemini, ses propres réponses ont le rôle « model ».
                role="user" if role == "user" else "model",
                parts=[types.Part(text=texte.strip()[:LONGUEUR_MAX_ENTREE])],
            )
        )
    return messages


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
        "gemini_configure": gemini_client is not None,
        "gemini_modele": MODELE_GEMINI,
        # Champs historiques conservés pour compatibilité avec les clients existants.
        "seq2seq": False,
        "seq2seq_epoch": None,
        "seq2seq_perte_validation": None,
        "seq2seq_erreur": None,
    })


@app.route("/predire", methods=["POST"])
def predire():
    texte, erreur = lire_texte()
    if erreur:
        return erreur

    X_phrase = vectorizer.transform([texte])
    probabilites = model.predict_proba(X_phrase)[0]
    indice = probabilites.argmax()
    intention = model.classes_[indice]
    confiance = float(probabilites[indice])
    repli = X_phrase.nnz == 0 or confiance < SEUIL_CONFIANCE
    reponse = random.choice(MESSAGES_INCOMPRIS if repli else reponses[intention])

    return jsonify({
        "texte_recu": texte,
        "intention": intention,
        "reponse": reponse,
        "confiance": round(confiance, 3),
        "repli": repli,
    })


@app.route("/discuter", methods=["POST"])
def discuter():
    texte, erreur = lire_texte()
    if erreur:
        return erreur

    if gemini_client is None:
        return jsonify({
            "erreur": "Gemini n'est pas configuré : la variable LLM_API_KEY est absente."
        }), 503

    donnees = request.get_json(silent=True) or {}
    temperature = donnees.get("temperature", TEMPERATURE_PAR_DEFAUT)
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not 0 <= temperature <= 2:
        return jsonify({"erreur": "temperature doit être un nombre entre 0 et 2"}), 400

    try:
        resultat = gemini_client.models.generate_content(
            model=MODELE_GEMINI,
            # Les échanges précédents (s'il y en a), puis la nouvelle question.
            contents=lire_historique(donnees)
            + [types.Content(role="user", parts=[types.Part(text=texte)])],
            config=types.GenerateContentConfig(
                temperature=float(temperature),
                # Les modèles Gemini 3 « réfléchissent » avant de répondre
                # (thinking). Utile pour un problème compliqué, mais beaucoup
                # trop lent pour une discussion : même « Merci » dépassait 20 s.
                # LOW = réflexion courte, réponse rapide. Réglable sans toucher
                # au code avec la variable Render GEMINI_THINKING (low, medium,
                # high ; minimal selon le modèle).
                thinking_config=types.ThinkingConfig(thinking_level=NIVEAU_REFLEXION),
                system_instruction=(
                    "Tu es Fatia, un assistant conversationnel utile, naturel et respectueux. "
                    "Réponds en français, sauf si l'utilisateur écrit clairement dans une autre langue. "
                    "Réponds directement à la question, avec des explications adaptées et sans inventer "
                    "des faits. Les derniers messages de la conversation peuvent t'être fournis "
                    "avant la question : sers-t'en pour comprendre la suite de l'échange. "
                    # L'app Nolia affiche du texte brut : sans cette consigne,
                    # Gemini écrit **gras**, # titres, etc., et les symboles
                    # apparaissent tels quels à l'écran.
                    "Écris en texte simple, sans aucune mise en forme Markdown : pas d'astérisques, "
                    "pas de dièses, pas de tableaux. Pour une liste, commence chaque ligne par un tiret. "
                    "Reste concis : quelques phrases suffisent, sauf si on te demande plus de détails."
                ),
            ),
        )
        reponse = (resultat.text or "").strip()
        if not reponse:
            return jsonify({"erreur": "Gemini n'a renvoyé aucun texte exploitable."}), 502

        return jsonify({
            "texte_recu": texte,
            "reponse": reponse,
            "repli": False,
            "tokens_inconnus": 0,
            "tokens_total": 0,
        })
    except Exception as exception:
        app.logger.exception("Erreur lors de l'appel à Gemini")
        return jsonify({
            "erreur": "Le service Gemini n'a pas pu générer de réponse.",
            "detail": f"{type(exception).__name__}: {exception}",
        }), 502


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
