# MiniML – Application de prédiction nutritionnelle

MiniML est une application web développée avec **FastAPI** permettant d'entraîner un modèle de Machine Learning et de prédire plusieurs indicateurs nutritionnels à partir de caractéristiques physico-chimiques.

---

## Fonctionnalités

- Modèle XGBoost multi-sorties  
- Encodage des variables catégorielles (OneHotEncoder)  
- Interface web FastAPI + Bootstrap  
- Prédictions nutritionnelles multi-critères  
- Page de performances du modèle  
- Historique des prédictions  
- Réentraînement dynamique  

---

## Structure du projet

├── main.py
├── models.py
├── Donnees_IA_2025.csv
├── templates/
├── requirements.txt
└── README.md


---

## Installation

Créer un environnement :

```bash
conda create -n miniml python=3.12
conda activate miniml
pip install -r requirements_API.txt
```


## Lancement de l'application : 
```bash
uvicorn main:app --reload
```



