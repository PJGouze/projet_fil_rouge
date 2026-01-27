from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import des données
def import_model():
    data_df = pd.read_csv("data/Donnees_IA_2025.csv", sep = ";", encoding="latin1")

    data_df= data_df.drop(columns=['Ordre', 'Code', 'Nom détaillé', 'Pays', 'Année récolte', 'Date mesure'])

    colonnes = data_df.columns
    colonnes_X = list(colonnes[:12])
    colonnes_Y = list(colonnes[12:])

    colonnes_cat = colonnes_X[:2]
    colonnes_num = colonnes_X[2:]


    categorical_cols = colonnes_cat
    numerical_cols = colonnes_num
    # OneHotEncoder
    enc = OneHotEncoder(sparse_output=False)
    X_cat = enc.fit_transform(data_df[categorical_cols])


    column_names = []
    for i, cat in enumerate(enc.categories_):
        column_names.extend([f"{categorical_cols[i]}_{c}" for c in cat])

    X_cat_df = pd.DataFrame(X_cat, columns=column_names)

    data_final = pd.concat([ X_cat_df, data_df[numerical_cols], data_df[colonnes_Y]], axis=1)

    # Colonnes à retirer pour X et à garder pour y
    colonnes_cibles = [
        'EB (kcal) kcal/kg brut', 
        'ED porc croissance (kcal) kcal/kg brut', 
        'EM porc croissance (kcal) kcal/kg brut', 
        'EN porc croissance (kcal) kcal/kg brut', 
        'EMAn coq (kcal) kcal/kg brut', 
        'EMAn poulet (kcal) kcal/kg brut', 
        'UFL 2018 par kg brut', 
        'UFV 2018 par kg brut', 
        'PDIA 2018 g_kg brut', 
        'PDI 2018 g_kg brut', 
        'BalProRu 2018 g_kg brut'
    ]

    # Génération de la variable cible
    y = data_final[colonnes_cibles]

    # Génération des descripteurs
    X = data_final.drop(columns=colonnes_cibles)

    cols_num = ['MS % brut', 'PB % brut', 'CB % brut', 'MGR % brut', 'MM % brut', 
                'NDF % brut', 'ADF % brut', 'Lignine % brut', 'Amidon % brut', 'Sucres % brut']

    for col in cols_num:
        X[col] = X[col].astype(str).str.replace(',', '.')
        X[col] = pd.to_numeric(X[col], errors='coerce') 

    for col in y.columns:
        y[col] = y[col].astype(str).str.strip().str.replace(',', '.')
        y[col] = pd.to_numeric(y[col], errors='coerce') 

    model = XGBRegressor()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)


    return model, enc, X.columns, X_train, X_test, y_train, y_test, mae, rmse, r2


def predict_from_input(model, encoder, X_columns, input_dict):
    df = pd.DataFrame([input_dict])

    # Séparer catégories et numériques
    categorical_cols = encoder.feature_names_in_.tolist()
    num_cols = [c for c in X_columns if c not in encoder.get_feature_names_out()]

    X_cat = encoder.transform(df[categorical_cols])
    X_cat_df = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out())

    X_num_df = df[num_cols]

    X_final = pd.concat([X_cat_df, X_num_df], axis=1)

    # Reorder columns
    X_final = X_final[X_columns]

    return model.predict(X_final)

