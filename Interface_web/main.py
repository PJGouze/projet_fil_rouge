from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import csv
import io

from models import import_model, predict_from_input


# ==========================
# Variables globales modèle
# ==========================
model = None
encoder = None
X_columns = None
X_train = None
X_test = None
y_train = None
y_test = None
mae = None
rmse = None
r2 = None

# ==============
# Target names
# ==============
TARGET_NAMES = [
    "EB (kcal) kcal/kg brut",
    "ED porc croissance (kcal) kcal/kg brut",
    "EM porc croissance (kcal) kcal/kg brut",
    "EN porc croissance (kcal) kcal/kg brut",
    "EMAn coq (kcal) kcal/kg brut",
    "EMAn poulet (kcal) kcal/kg brut",
    "UFL 2018 par kg brut",
    "UFV 2018 par kg brut",
    "PDIA 2018 g/kg brut",
    "PDI 2018 g/kg brut",
    "BalProRu 2018 g/kg brut"
]

# ==============
# Chargement du modèle
# ==============

def load_model():
    global model, encoder, X_columns, X_train, X_test, y_train, y_test, mae, rmse, r2
    model, encoder, X_columns, X_train, X_test, y_train, y_test, mae, rmse, r2 = import_model()


# Chargement initial
load_model()


# ==========================
# FastAPI
# ==========================
app = FastAPI(title="Mini calculatrice FastAPI")
templates = Jinja2Templates(directory="templates")

prediction_history = []



# ==========================
# Routes
# ==========================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": None,
        "page": "home"
    })


@app.post("/calcul", response_class=HTMLResponse)
def calcul(
    request: Request,
    x: float = Form(...),
    y: float = Form(...),
    operation: str = Form(...)
):
    if operation == "add":
        result = x + y
        op_symbol = "+"
    elif operation == "mul":
        result = x * y
        op_symbol = "*"
    else:
        result = "Opération invalide"
        op_symbol = "?"

    history.append({
        "x": x,
        "y": y,
        "operation": op_symbol,
        "result": result
    })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "page": "home"
    })


@app.get("/history", response_class=HTMLResponse)
def show_history(request: Request):
    return templates.TemplateResponse("history.html", {
        "request": request,
        "prediction_history": prediction_history,
        "target_names": TARGET_NAMES,
        "page": "history"
    })


@app.get("/model_info", response_class=HTMLResponse)
def model_info(request: Request):
    return templates.TemplateResponse("model_info.html", {
        "request": request,
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "n_features": X_train.shape[1],
        "model_name": type(model).__name__,
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 3),
        "params": model.get_params(),
        "page": "model"
    })


@app.post("/retrain")
def retrain_model():
    load_model()
    return RedirectResponse(url="/model_info", status_code=303)


@app.get("/predict", response_class=HTMLResponse)
def predict_form(request: Request):
    categories = {col: encoder.categories_[i].tolist()
                  for i, col in enumerate(encoder.feature_names_in_)}

    numeric_cols = [c for c in X_columns if c not in encoder.get_feature_names_out()]

    return templates.TemplateResponse("predict.html", {
        "request": request,
        "categories": categories,
        "numeric_cols": numeric_cols,
        "page": "predict"
    })
@app.post("/predict", response_class=HTMLResponse)
async def predict_result(request: Request):

    form = await request.form()
    form_data = dict(form)

    numeric_cols = [c for c in X_columns if c not in encoder.get_feature_names_out()]

    cleaned_data = {}

    try:
        for col, val in form_data.items():
            if col in numeric_cols:
                cleaned_data[col] = float(val)
            else:
                cleaned_data[col] = val
    except ValueError:
        categories = {col: encoder.categories_[i].tolist()
                      for i, col in enumerate(encoder.feature_names_in_)}

        return templates.TemplateResponse("predict.html", {
            "request": request,
            "categories": categories,
            "numeric_cols": numeric_cols,
            "error": "Valeurs non conformes : les champs numériques doivent être des entiers ou des flottants.",
            "page": "predict"
        })

    y_pred = predict_from_input(model, encoder, X_columns, cleaned_data)[0]

    results = [
        {"name": name, "value": round(float(val), 2)}
        for name, val in zip(TARGET_NAMES, y_pred)
    ]
    prediction_history.append(results)

    categories = {col: encoder.categories_[i].tolist()
                  for i, col in enumerate(encoder.feature_names_in_)}

    return templates.TemplateResponse("predict.html", {
        "request": request,
        "categories": categories,
        "numeric_cols": numeric_cols,
        "results": results,
        "page": "predict"
    })

