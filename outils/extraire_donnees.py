"""Extrait le dataset d'intentions depuis essai.ipynb vers data/intentions.json.

Lit directement le source des cellules (aucun unpickling, donc aucune dépendance
à joblib ou scikit-learn). Le notebook contient plusieurs versions successives des
dictionnaires : on retient la dernière de chaque, pas un numéro de cellule figé.
"""

import ast
import json
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
NOTEBOOK = RACINE / "essai.ipynb"
SORTIE = RACINE / "data" / "intentions.json"


def derniere_affectation(cellules, nom):
    """Renvoie (index_cellule, dict) de la dernière affectation `nom = {...}`."""
    trouve = None
    for index, cellule in enumerate(cellules):
        if cellule.get("cell_type") != "code":
            continue
        source = "".join(cellule["source"])
        if f"{nom} = {{" not in source:
            continue
        try:
            arbre = ast.parse(source)
        except SyntaxError:
            continue
        for noeud in arbre.body:
            if not isinstance(noeud, ast.Assign) or not isinstance(noeud.value, ast.Dict):
                continue
            cibles = [c.id for c in noeud.targets if isinstance(c, ast.Name)]
            if nom in cibles:
                trouve = (index, ast.literal_eval(noeud.value))
    return trouve


def main():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cellules = notebook["cells"]
    print(f"Notebook lu : {len(cellules)} cellules")

    resultats = {}
    for nom in ("exemples", "reponses"):
        trouve = derniere_affectation(cellules, nom)
        if trouve is None:
            sys.exit(f"ERREUR : aucune affectation `{nom} = {{...}}` trouvée dans le notebook.")
        index, valeur = trouve
        total = sum(len(v) for v in valeur.values())
        print(f"  {nom:10} → cellule {index} : {len(valeur)} intentions, {total} entrées")
        resultats[nom] = valeur

    exemples, reponses = resultats["exemples"], resultats["reponses"]

    sans_reponses = sorted(set(exemples) - set(reponses))
    sans_exemples = sorted(set(reponses) - set(exemples))
    if sans_reponses or sans_exemples:
        sys.exit(
            "ERREUR : les intentions ne correspondent pas entre les deux dictionnaires.\n"
            f"  Sans réponses : {sans_reponses}\n"
            f"  Sans exemples : {sans_exemples}"
        )

    dataset = {
        intention: {"exemples": exemples[intention], "reponses": reponses[intention]}
        for intention in exemples
    }

    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    SORTIE.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\nÉcrit dans {SORTIE.relative_to(RACINE)}")
    for intention, contenu in dataset.items():
        print(f"  {intention:16} {len(contenu['exemples']):3} exemples, "
              f"{len(contenu['reponses'])} réponses")


if __name__ == "__main__":
    main()
