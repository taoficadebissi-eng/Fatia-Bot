"""Inférence : charge un checkpoint entraîné et génère une réponse token par token."""

from pathlib import Path

import torch

from fatia.donnees import Vocabulaire, encoder_phrase
from fatia.entrainement import NOM_POIDS, NOM_VOCABULAIRE
from fatia.modele import Seq2Seq

MESSAGE_REPLI = "Désolé, je ne connais aucun des mots que tu viens d'employer."


class Repondeur:
    """Modèle chargé en mémoire, prêt à répondre."""

    def __init__(self, dossier_modele):
        dossier = Path(dossier_modele)
        self.vocabulaire = Vocabulaire.charger(dossier / NOM_VOCABULAIRE)

        checkpoint = torch.load(dossier / NOM_POIDS, map_location="cpu", weights_only=True)
        self.modele = Seq2Seq(**checkpoint["hyperparametres"])
        self.modele.load_state_dict(checkpoint["etat_modele"])
        self.modele.eval()

        self.epoch = checkpoint["epoch"]
        self.perte_validation = checkpoint["perte_validation"]

    def repondre(self, phrase, temperature=0.0, longueur_max=30):
        """Renvoie un dict : réponse générée + diagnostic sur les mots inconnus.

        temperature = 0 → décodage glouton (déterministe) ; > 0 → échantillonnage.
        """
        identifiants, nb_inconnus, nb_tokens = encoder_phrase(self.vocabulaire, phrase)

        if nb_tokens == 0 or nb_inconnus == nb_tokens:
            return {
                "reponse": MESSAGE_REPLI,
                "repli": True,
                "tokens_inconnus": nb_inconnus,
                "tokens_total": nb_tokens,
            }

        entrees = torch.tensor([identifiants], dtype=torch.long)
        longueurs = torch.tensor([len(identifiants)], dtype=torch.long)
        tokens_generes = []

        with torch.no_grad():
            etat = self.modele.encodeur(entrees, longueurs)
            token = torch.tensor([self.vocabulaire.sos], dtype=torch.long)

            for _ in range(longueur_max):
                logits, etat = self.modele.decodeur(token, etat)

                if temperature > 0:
                    probabilites = torch.softmax(logits.squeeze(0) / temperature, dim=-1)
                    suivant = int(torch.multinomial(probabilites, 1))
                else:
                    suivant = int(logits.argmax(dim=1))

                if suivant == self.vocabulaire.eos:
                    break

                tokens_generes.append(suivant)
                token = torch.tensor([suivant], dtype=torch.long)

        return {
            "reponse": _recoller(self.vocabulaire.decoder(tokens_generes)),
            "repli": False,
            "tokens_inconnus": nb_inconnus,
            "tokens_total": nb_tokens,
        }


def _recoller(tokens):
    """Reconstruit une phrase lisible en recollant la ponctuation."""
    texte = ""
    for token in tokens:
        if not texte:
            texte = token
        elif token in ",.!?;:'-" or texte.endswith(("'", "-")):
            texte += token
        else:
            texte += " " + token
    return texte[:1].upper() + texte[1:]
