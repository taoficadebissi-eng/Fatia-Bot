"""Pipeline de données du seq2seq : dataset → paires question/réponse → vocabulaire → identifiants.

Ce module n'utilise que la bibliothèque standard (pas de torch) : la conversion en
tenseurs est faite côté entraînement. Ça permet de tester toute la chaîne de données
sans dépendance ML installée.
"""

import json
import random
import re
from pathlib import Path

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
TOKENS_SPECIAUX = (PAD, SOS, EOS, UNK)

_MOTIF_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokeniser(phrase):
    """Découpe en mots minuscules, la ponctuation devenant des tokens à part entière."""
    return _MOTIF_TOKEN.findall(phrase.lower())


def charger_dataset(chemin):
    return json.loads(Path(chemin).read_text(encoding="utf-8"))


def separer_exemples(dataset, proportion_validation=0.2, graine=42):
    """Sépare les exemples en train/validation AVANT le produit croisé.

    Le split se fait au niveau des exemples (et par intention), pas au niveau des
    paires : sinon une même question se retrouverait des deux côtés une fois par
    réponse de son intention, la loss de validation serait artificiellement basse
    et ne détecterait plus le sur-apprentissage.
    """
    alea = random.Random(graine)
    train, validation = {}, {}

    for intention, contenu in dataset.items():
        exemples = list(contenu["exemples"])
        alea.shuffle(exemples)

        nb_validation = max(1, round(len(exemples) * proportion_validation))
        nb_validation = min(nb_validation, len(exemples) - 1)

        validation[intention] = {
            "exemples": exemples[:nb_validation],
            "reponses": contenu["reponses"],
        }
        train[intention] = {
            "exemples": exemples[nb_validation:],
            "reponses": contenu["reponses"],
        }

    return train, validation


def construire_paires(dataset):
    """Produit croisé exemples × réponses au sein de chaque intention."""
    paires = []
    for contenu in dataset.values():
        for exemple in contenu["exemples"]:
            for reponse in contenu["reponses"]:
                paires.append((exemple, reponse))
    return paires


class Vocabulaire:
    """Vocabulaire partagé entrée/sortie, construit sur les données d'entraînement seules."""

    def __init__(self, tokens):
        self.index_vers_token = list(tokens)
        self.token_vers_index = {token: i for i, token in enumerate(self.index_vers_token)}
        self.pad = self.token_vers_index[PAD]
        self.sos = self.token_vers_index[SOS]
        self.eos = self.token_vers_index[EOS]
        self.unk = self.token_vers_index[UNK]

    def __len__(self):
        return len(self.index_vers_token)

    @classmethod
    def construire(cls, paires, frequence_min=1):
        frequences = {}
        for question, reponse in paires:
            for token in tokeniser(question) + tokeniser(reponse):
                frequences[token] = frequences.get(token, 0) + 1

        retenus = sorted(t for t, n in frequences.items() if n >= frequence_min)
        return cls(list(TOKENS_SPECIAUX) + retenus)

    def encoder(self, tokens):
        return [self.token_vers_index.get(token, self.unk) for token in tokens]

    def decoder(self, identifiants, ignorer_speciaux=True):
        speciaux = {self.pad, self.sos, self.eos}
        return [
            self.index_vers_token[i]
            for i in identifiants
            if not (ignorer_speciaux and i in speciaux)
        ]

    def sauvegarder(self, chemin):
        chemin = Path(chemin)
        chemin.parent.mkdir(parents=True, exist_ok=True)
        chemin.write_text(
            json.dumps(self.index_vers_token, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def charger(cls, chemin):
        return cls(json.loads(Path(chemin).read_text(encoding="utf-8")))


def encoder_phrase(vocabulaire, phrase):
    """Encode une phrase d'entrée. Renvoie (identifiants, nb_inconnus, nb_tokens).

    Les tokens hors-vocabulaire deviennent <unk> au lieu de lever une exception.
    Un appelant peut comparer nb_inconnus et nb_tokens : quand tous les tokens sont
    inconnus, générer une réponse n'a aucun sens et mieux vaut un message de repli.
    """
    tokens = tokeniser(phrase)
    identifiants = vocabulaire.encoder(tokens)
    nb_inconnus = sum(1 for i in identifiants if i == vocabulaire.unk)
    return identifiants + [vocabulaire.eos], nb_inconnus, len(tokens)


def encoder_paires(vocabulaire, paires):
    """Encode les paires en (entrée + <eos>, <sos> + cible + <eos>)."""
    encodees = []
    for question, reponse in paires:
        entree = vocabulaire.encoder(tokeniser(question)) + [vocabulaire.eos]
        cible = [vocabulaire.sos] + vocabulaire.encoder(tokeniser(reponse)) + [vocabulaire.eos]
        encodees.append((entree, cible))
    return encodees


def preparer_paires_brutes(chemin):
    """Charge un fichier de paires [question, réponse] déjà appariées à la main.

    Sert au test de sur-apprentissage : train et validation sont volontairement
    identiques, puisqu'on cherche justement à vérifier que le modèle mémorise.
    """
    paires = [tuple(paire) for paire in json.loads(Path(chemin).read_text(encoding="utf-8"))]
    vocabulaire = Vocabulaire.construire(paires)
    encodees = encoder_paires(vocabulaire, paires)

    return {
        "vocabulaire": vocabulaire,
        "train": encodees,
        "validation": encodees,
        "paires_train": paires,
        "paires_validation": paires,
    }


def preparer(chemin_dataset, proportion_validation=0.2, graine=42):
    """Chaîne complète : dataset → paires train/val encodées + vocabulaire."""
    dataset = charger_dataset(chemin_dataset)
    dataset_train, dataset_validation = separer_exemples(
        dataset, proportion_validation, graine
    )

    paires_train = construire_paires(dataset_train)
    paires_validation = construire_paires(dataset_validation)

    vocabulaire = Vocabulaire.construire(paires_train)

    return {
        "vocabulaire": vocabulaire,
        "train": encoder_paires(vocabulaire, paires_train),
        "validation": encoder_paires(vocabulaire, paires_validation),
        "paires_train": paires_train,
        "paires_validation": paires_validation,
    }
