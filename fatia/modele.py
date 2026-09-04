"""Encodeur-décodeur GRU (seq2seq), sans attention.

L'encodeur compresse la question en un unique vecteur d'état ; le décodeur génère la
réponse token par token à partir de cet état.
"""

import random

import torch
from torch import nn


class Encodeur(nn.Module):
    def __init__(self, taille_vocabulaire, dim_embedding=128, dim_cachee=256, index_pad=0):
        super().__init__()
        self.embedding = nn.Embedding(taille_vocabulaire, dim_embedding, padding_idx=index_pad)
        self.gru = nn.GRU(dim_embedding, dim_cachee, batch_first=True)

    def forward(self, entrees, longueurs=None):
        """entrees : (batch, longueur) → état caché (1, batch, dim_cachee).

        `longueurs` donne le nombre de tokens réels de chaque ligne. Sans elle, le GRU
        consommerait aussi les `<pad>` et l'état final dépendrait du remplissage : une
        phrase courte n'aurait alors pas le même encodage à l'entraînement (paddée dans
        son batch) et à l'inférence (seule, non paddée).
        """
        embarques = self.embedding(entrees)

        if longueurs is None:
            _, etat = self.gru(embarques)
            return etat

        paquet = nn.utils.rnn.pack_padded_sequence(
            embarques, longueurs.cpu(), batch_first=True, enforce_sorted=False
        )
        _, etat = self.gru(paquet)
        return etat


class Decodeur(nn.Module):
    def __init__(self, taille_vocabulaire, dim_embedding=128, dim_cachee=256, index_pad=0):
        super().__init__()
        self.embedding = nn.Embedding(taille_vocabulaire, dim_embedding, padding_idx=index_pad)
        self.gru = nn.GRU(dim_embedding, dim_cachee, batch_first=True)
        self.sortie = nn.Linear(dim_cachee, taille_vocabulaire)

    def forward(self, tokens, etat):
        """tokens : (batch,) → logits (batch, vocabulaire) et nouvel état."""
        sorties, etat = self.gru(self.embedding(tokens).unsqueeze(1), etat)
        return self.sortie(sorties.squeeze(1)), etat


class Seq2Seq(nn.Module):
    def __init__(self, taille_vocabulaire, dim_embedding=128, dim_cachee=256, index_pad=0):
        super().__init__()
        self.taille_vocabulaire = taille_vocabulaire
        self.encodeur = Encodeur(taille_vocabulaire, dim_embedding, dim_cachee, index_pad)
        self.decodeur = Decodeur(taille_vocabulaire, dim_embedding, dim_cachee, index_pad)
        self.hyperparametres = {
            "taille_vocabulaire": taille_vocabulaire,
            "dim_embedding": dim_embedding,
            "dim_cachee": dim_cachee,
            "index_pad": index_pad,
        }

    def forward(self, entrees, cibles, longueurs=None, taux_teacher_forcing=0.5):
        """entrees (batch, L_e), cibles (batch, L_c) commençant par <sos>.

        Renvoie les logits (batch, L_c - 1, vocabulaire) : à chaque pas on prédit le
        token suivant de la cible.
        """
        batch, longueur_cible = cibles.shape
        etat = self.encodeur(entrees, longueurs)
        token = cibles[:, 0]

        logits = torch.zeros(
            batch, longueur_cible - 1, self.taille_vocabulaire, device=entrees.device
        )

        for pas in range(longueur_cible - 1):
            logits_pas, etat = self.decodeur(token, etat)
            logits[:, pas] = logits_pas

            if random.random() < taux_teacher_forcing:
                token = cibles[:, pas + 1]
            else:
                token = logits_pas.argmax(dim=1)

        return logits
