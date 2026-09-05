"""Entraîne le classifieur d'intentions à partir de data/intentions.json.

Remplace les cellules 42-48 de essai.ipynb : l'entraînement ne dépend plus de
l'exécution manuelle du notebook dans le bon ordre.

    python entrainer_classifieur.py                      # écrit à la racine
    python entrainer_classifieur.py --sortie /tmp/essai   # écrit ailleurs

Trois choix de vectorisation, chacun réglant un défaut observé en production :

- `strip_accents="unicode"` : « ça va » et « ca va » deviennent identiques, car
  les utilisateurs tapent souvent sans accents sur mobile.
- `token_pattern` gardant « ? » et « ! » : sans eux, « ça va bien ? » (une
  question) et « oui ça va bien » (une réponse) ont presque le même sac de mots,
  et le bot repose sa question en boucle.
- `ngram_range=(1, 2)` : les bigrammes « va ? » ou « oui ça » portent le signal
  question/réponse qu'un mot isolé ne peut pas porter.
"""

import argparse
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

RACINE = Path(__file__).resolve().parent
MOTIF_TOKEN = r"(?u)\b\w+\b|[?!]"


def charger(chemin):
    dataset = json.loads(Path(chemin).read_text(encoding="utf-8"))
    textes, intentions = [], []
    for intention, contenu in dataset.items():
        for exemple in contenu["exemples"]:
            textes.append(exemple)
            intentions.append(intention)
    reponses = {i: c["reponses"] for i, c in dataset.items()}
    return textes, intentions, reponses


def construire_vectorizer():
    return CountVectorizer(
        strip_accents="unicode",
        token_pattern=MOTIF_TOKEN,
        ngram_range=(1, 2),
        lowercase=True,
    )


def main():
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--dataset", default=RACINE / "data" / "intentions.json")
    parseur.add_argument("--sortie", default=RACINE)
    parseur.add_argument("--test", type=float, default=0.2)
    parseur.add_argument("--graine", type=int, default=42)
    args = parseur.parse_args()

    textes, intentions, reponses = charger(args.dataset)
    print(f"{len(textes)} exemples, {len(set(intentions))} intentions\n")

    textes_train, textes_test, y_train, y_test = train_test_split(
        textes, intentions, test_size=args.test,
        random_state=args.graine, stratify=intentions,
    )

    # Le vectorizer n'apprend son vocabulaire que sur le train : sinon les mots
    # vus uniquement en test gonfleraient artificiellement le score.
    vectorizer = construire_vectorizer()
    X_train = vectorizer.fit_transform(textes_train)
    X_test = vectorizer.transform(textes_test)

    # class_weight equilibre les intentions : salutation a 4x plus d'exemples
    # qu'excuse, et sans cela le modele s'y rabat quand il hesite.
    modele = LogisticRegression(max_iter=1000, class_weight="balanced")
    modele.fit(X_train, y_train)

    predictions = modele.predict(X_test)
    print(f"Vocabulaire : {len(vectorizer.get_feature_names_out())} traits "
          f"(unigrammes + bigrammes)")
    print(f"Exactitude sur le test : {accuracy_score(y_test, predictions):.1%}\n")
    print(classification_report(y_test, predictions, zero_division=0))

    # Le modele livre est reentraine sur la totalite des donnees : le split
    # ci-dessus ne sert qu'a mesurer honnetement, pas a jeter 20% des exemples.
    vectorizer_final = construire_vectorizer()
    X_tout = vectorizer_final.fit_transform(textes)
    modele_final = LogisticRegression(max_iter=1000, class_weight="balanced")
    modele_final.fit(X_tout, intentions)

    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    joblib.dump(modele_final, sortie / "modele_intention.pkl")
    joblib.dump(vectorizer_final, sortie / "vectorizer_intention.pkl")
    joblib.dump(reponses, sortie / "reponses_intention.pkl")
    print(f"Modèle réentraîné sur les {len(textes)} exemples et écrit dans {sortie}/")


if __name__ == "__main__":
    main()
