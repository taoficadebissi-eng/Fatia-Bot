"""Boucle d'entraînement du seq2seq, avec conservation du meilleur checkpoint."""

import json
import math
import random
from pathlib import Path

import torch
from torch import nn

from fatia.modele import Seq2Seq

NOM_POIDS = "seq2seq.pt"
NOM_VOCABULAIRE = "vocabulaire.json"


def creer_batches(paires_encodees, index_pad, taille_batch, melanger=True, graine=None):
    """Découpe en batches, applique le padding, et renvoie les longueurs réelles.

    Les longueurs servent à faire ignorer le padding au GRU de l'encodeur : sans elles,
    l'encodage d'une phrase dépendrait du nombre de `<pad>` de son batch.
    """
    paires = list(paires_encodees)
    if melanger:
        random.Random(graine).shuffle(paires)

    for debut in range(0, len(paires), taille_batch):
        lot = paires[debut : debut + taille_batch]
        longueur_entree = max(len(entree) for entree, _ in lot)
        longueur_cible = max(len(cible) for _, cible in lot)

        entrees = torch.full((len(lot), longueur_entree), index_pad, dtype=torch.long)
        cibles = torch.full((len(lot), longueur_cible), index_pad, dtype=torch.long)
        longueurs = torch.tensor([len(entree) for entree, _ in lot], dtype=torch.long)

        for ligne, (entree, cible) in enumerate(lot):
            entrees[ligne, : len(entree)] = torch.tensor(entree, dtype=torch.long)
            cibles[ligne, : len(cible)] = torch.tensor(cible, dtype=torch.long)

        yield entrees, longueurs, cibles


def _perte_batch(modele, critere, entrees, longueurs, cibles, taux_teacher_forcing):
    logits = modele(entrees, cibles, longueurs, taux_teacher_forcing)
    return critere(logits.reshape(-1, logits.size(-1)), cibles[:, 1:].reshape(-1))


def evaluer(modele, paires, critere, index_pad, taille_batch):
    modele.eval()
    total, nb_batches = 0.0, 0
    with torch.no_grad():
        for entrees, longueurs, cibles in creer_batches(
            paires, index_pad, taille_batch, melanger=False
        ):
            # Sans teacher forcing : le modèle se relit lui-même, comme à l'inférence.
            total += _perte_batch(modele, critere, entrees, longueurs, cibles, 0.0).item()
            nb_batches += 1
    return total / max(nb_batches, 1)


def entrainer(
    donnees,
    dossier_sortie,
    epochs=100,
    taille_batch=32,
    taux_apprentissage=1e-3,
    taux_teacher_forcing=0.5,
    dim_embedding=128,
    dim_cachee=256,
    clip_gradient=1.0,
    graine=42,
):
    torch.manual_seed(graine)
    random.seed(graine)

    vocabulaire = donnees["vocabulaire"]
    dossier_sortie = Path(dossier_sortie)
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    vocabulaire.sauvegarder(dossier_sortie / NOM_VOCABULAIRE)

    modele = Seq2Seq(len(vocabulaire), dim_embedding, dim_cachee, vocabulaire.pad)
    optimiseur = torch.optim.Adam(modele.parameters(), lr=taux_apprentissage)
    critere = nn.CrossEntropyLoss(ignore_index=vocabulaire.pad)

    meilleure_perte = math.inf
    meilleur_epoch = 0
    historique = []

    for epoch in range(1, epochs + 1):
        modele.train()
        total, nb_batches = 0.0, 0

        for entrees, longueurs, cibles in creer_batches(
            donnees["train"], vocabulaire.pad, taille_batch, graine=graine + epoch
        ):
            optimiseur.zero_grad()
            perte = _perte_batch(
                modele, critere, entrees, longueurs, cibles, taux_teacher_forcing
            )
            perte.backward()
            nn.utils.clip_grad_norm_(modele.parameters(), clip_gradient)
            optimiseur.step()

            total += perte.item()
            nb_batches += 1

        perte_train = total / max(nb_batches, 1)
        perte_validation = evaluer(
            modele, donnees["validation"], critere, vocabulaire.pad, taille_batch
        )
        historique.append({
            "epoch": epoch,
            "perte_train": perte_train,
            "perte_validation": perte_validation,
        })

        marqueur = ""
        if perte_validation < meilleure_perte:
            meilleure_perte = perte_validation
            meilleur_epoch = epoch
            torch.save(
                {
                    "etat_modele": modele.state_dict(),
                    "hyperparametres": modele.hyperparametres,
                    "epoch": epoch,
                    "perte_validation": perte_validation,
                },
                dossier_sortie / NOM_POIDS,
            )
            marqueur = "  <- sauvegardé"

        print(
            f"epoch {epoch:4}/{epochs}  "
            f"train {perte_train:.4f} (ppl {math.exp(min(perte_train, 20)):7.2f})  "
            f"val {perte_validation:.4f} (ppl {math.exp(min(perte_validation, 20)):7.2f})"
            f"{marqueur}"
        )

    (dossier_sortie / "historique.json").write_text(
        json.dumps(historique, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"\nMeilleur checkpoint : epoch {meilleur_epoch}, "
        f"perte de validation {meilleure_perte:.4f}"
    )
    print(f"Écrit dans {dossier_sortie}/ ({NOM_POIDS}, {NOM_VOCABULAIRE}, historique.json)")

    return historique
