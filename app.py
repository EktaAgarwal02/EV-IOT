from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import joblib
import os


app = Flask(__name__)
app.secret_key = "dev-secret-key"  # for flash messages


# Load model and training metadata once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgboost.pkl")
DATA_PATH = os.path.join(os.path.dirname(__file__), "EV_Predictive_Maintenance_Dataset_15min.csv")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Failed to load model: {e}")

try:
    base_df = pd.read_csv(DATA_PATH)
except Exception as e:
    base_df = None
    print(f"Failed to load base data: {e}")


def build_timestamp_mapping(df: pd.DataFrame) -> dict:
    if "Timestamp" not in df.columns:
        return {}
    # Recreate a stable mapping similar to LabelEncoder fitted on training timestamps
    unique_vals = pd.Index(df["Timestamp"].astype(str).unique())
    return {val: idx for idx, val in enumerate(unique_vals)}


# Prepare training feature columns and timestamp label mapping
if base_df is not None:
    TRAIN_FEATURES = [c for c in base_df.columns if c not in ["Failure_Binary", "Failure_Probability"]]
    TIMESTAMP_MAP = build_timestamp_mapping(base_df)
    # Compute feature importances if available, fallback to order
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is not None and len(importances) == len(TRAIN_FEATURES):
            feat_imp = sorted(zip(TRAIN_FEATURES, importances), key=lambda x: x[1], reverse=True)
            top = [f for f, _ in feat_imp[:10]]  # top 10
        else:
            top = TRAIN_FEATURES[:10]
    except Exception:
        top = TRAIN_FEATURES[:10]

    # Always include these in the form if present
    always = [f for f in ["Timestamp", "Maintenance_Type"] if f in TRAIN_FEATURES]
    FORM_FEATURES = []
    for f in always + top:
        if f not in FORM_FEATURES:
            FORM_FEATURES.append(f)
else:
    TRAIN_FEATURES = []
    TIMESTAMP_MAP = {}
    FORM_FEATURES = []


def preprocess_input(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    # Drop target columns if they exist
    for col in ["Failure_Binary", "Failure_Probability"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Ensure Timestamp numeric encoding consistent with training mapping
    if "Timestamp" in df.columns:
        col = df["Timestamp"].astype(str)
        df["Timestamp"] = col.map(TIMESTAMP_MAP).fillna(-1).astype(int)

    # Add any missing training features with 0, and keep only features used in training
    for feat in TRAIN_FEATURES:
        if feat not in df.columns:
            fill_val = 0
            if base_df is not None and feat in base_df.columns:
                try:
                    fill_val = float(base_df[feat].median())
                except Exception:
                    fill_val = 0
            df[feat] = fill_val
    df = df[TRAIN_FEATURES]

    # Coerce to numeric types where possible
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df


@app.route("/")
def index():
    # Use existing index.html if present; otherwise fall back to predict page
    try:
        return render_template("index.html")
    except Exception:
        return redirect(url_for("predict"))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", table_html=None, metrics=None)

    # POST: handle file upload and run predictions
    if model is None or base_df is None:
        flash("Model or base dataset not available. Please check server logs.")
        return redirect(url_for("predict"))

    file = request.files.get("file")
    if not file or file.filename == "":
        flash("Please select a CSV file to upload.")
        return redirect(url_for("predict"))

    try:
        input_df = pd.read_csv(file)
    except Exception as e:
        flash(f"Failed to read CSV: {e}")
        return redirect(url_for("predict"))

    # Keep a copy for display
    display_df = input_df.copy()

    # Preprocess
    X = preprocess_input(input_df)

    # Predict
    try:
        y_pred = model.predict(X)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
        else:
            # fallback if proba not available
            y_proba = np.zeros(len(y_pred), dtype=float)
    except Exception as e:
        flash(f"Prediction failed: {e}")
        return redirect(url_for("predict"))

    # Attach results for preview
    result_df = display_df.copy()
    result_df["Prediction"] = y_pred
    result_df["Failure_Probability_Pred"] = y_proba

    # Simple metrics summary
    metrics = {
        "rows": int(len(result_df)),
        "positives": int((y_pred == 1).sum()),
        "avg_probability": float(np.mean(y_proba)) if len(y_proba) else 0.0,
    }

    # Show only first 50 rows to keep UI light
    preview = result_df.head(50)
    table_html = preview.to_html(classes="table table-striped table-sm", index=False)

    return render_template("predict.html", table_html=table_html, metrics=metrics)


@app.route("/form", methods=["GET", "POST"])
def form_predict():
    if model is None or base_df is None:
        flash("Model or base dataset not available. Please check server logs.")
        return redirect(url_for("index"))

    # Build simple schema from base data types for inputs
    dtypes = {}
    defaults = {}
    options = {}
    for col in FORM_FEATURES:
        if col == "Timestamp":
            dtypes[col] = "text"
            defaults[col] = ""
        else:
            kind = str(base_df[col].dtype)
            dtypes[col] = "number" if kind.startswith(("float", "int")) else "text"
            try:
                defaults[col] = float(base_df[col].median())
            except Exception:
                defaults[col] = 0

        # Provide dropdown options for small-cardinality integer-like columns (e.g., Maintenance_Type)
        if col in base_df.columns:
            uniques = sorted(pd.Series(base_df[col]).dropna().unique().tolist())
            if len(uniques) > 0 and len(uniques) <= 6 and all(isinstance(x, (int, np.integer)) for x in uniques):
                options[col] = uniques

    if request.method == "GET":
        return render_template("form_predict.html", feature_types=dtypes, defaults=defaults, options=options, result=None)

    # POST: read form values into a single-row DataFrame
    row = {}
    for feat in FORM_FEATURES:
        val = request.form.get(feat, "")
        row[feat] = val

    input_df = pd.DataFrame([row])
    X = preprocess_input(input_df)

    try:
        y_pred = model.predict(X)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
        else:
            y_proba = np.zeros(len(y_pred), dtype=float)
    except Exception as e:
        flash(f"Prediction failed: {e}")
        return redirect(url_for("form_predict"))

    result = {
        "prediction": int(y_pred[0]) if len(y_pred) else None,
        "probability": float(y_proba[0]) if len(y_proba) else 0.0,
    }

    return render_template("form_predict.html", feature_types=dtypes, defaults=defaults, options=options, result=result)

if __name__ == "__main__":
    app.run(debug=True)