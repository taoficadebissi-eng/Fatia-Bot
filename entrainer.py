"""Point d'entrée d'entraînement du seq2seq.

Entraînement normal :
    python entrainer.py --intentions data/intentions.json --sortie modeles --epochs 100

Test de sur-apprentissage (sortie isolée pour ne pas écraser le modèle principal) :
    python entrainer.py --paires data/paires_test.json \
        --sortie modeles/test_surapprentissage --epochs 400
"""

import argparse

from fatia import donnees
from fatia.entrainement import entrainer


def main():
    parseur = argparse.ArgumentParser(description=__doc__)
    source = parseur.add_mutually_exclusive_group(required=True)
    source.add_argument("--intentions", help="dataset d'intentions (produit croisé Q×R)")
    source.add_argument("--paires", help="paires [question, réponse] déjà appariées")

    parseur.add_argument("--sortie", required=True, help="dossier des poids et du vocabulaire")
    parseur.add_argument("--epochs", type=int, default=100)
    parseur.add_argument("--batch", type=int, default=32)
    parseur.add_argument("--lr", type=float, default=1e-3)
    parseur.add_argument("--teacher-forcing", type=float, default=0.5)
    parseur.add_argument("--validation", type=float, default=0.2)
    parseur.add_argument("--graine", type=int, default=42)
    args = parseur.parse_args()

    if args.paires:
        preparees = donnees.preparer_paires_brutes(args.paires)
        print(f"{len(preparees['paires_train'])} paires appariées à la main "
              f"(train = validation : test de mémorisation)")
    else:
        preparees = donnees.preparer(args.intentions, args.validation, args.graine)
        print(
            f"{len(preparees['train'])} paires d'entraînement, "
            f"{len(preparees['validation'])} de validation "
            f"(split au niveau des exemples)"
        )

    print(f"Vocabulaire : {len(preparees['vocabulaire'])} tokens\n")

    entrainer(
        preparees,
        dossier_sortie=args.sortie,
        epochs=args.epochs,
        taille_batch=args.batch,
        taux_apprentissage=args.lr,
        taux_teacher_forcing=args.teacher_forcing,
        graine=args.graine,
    )


if __name__ == "__main__":
    main()
