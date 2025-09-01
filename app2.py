# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# -------------------------------
# Page settings
# -------------------------------
st.set_page_config(page_title="⚡ EV Predictive Maintenance Suite", layout="wide")
st.title("⚡ EV Predictive Maintenance Suite")
st.caption("Multiclass failure prediction • Remaining Useful Life (RUL) • Health score • Explainability")

# -------------------------------
# Helpers
# -------------------------------
@st.cache_data
def load_pickle(path):
    return joblib.load(path)

def try_load_feature_list(path, fallback_names=None):
    """Load feature list from pkl if present; otherwise use fallback (e.g., model feature names)."""
    if Path(path).exists():
        return joblib.load(path)
    return list(fallback_names) if fallback_names is not None else None

def ensure_dataframe_order(input_dict, feature_list):
    """Build a 1-row DataFrame in exactly the order the model expects."""
    row = {k: input_dict.get(k, 0) for k in feature_list}
    return pd.DataFrame([row], columns=feature_list)

# Label mapping for your multiclass classifier
CLASS_LABELS = {
    0: "✅ Healthy",
    1: "⚠️ Battery Issue",
    2: "⚠️ Motor Issue",
    3: "⚠️ Brake Issue",
    4: "⚠️ Tire/Suspension Issue",
}

# Categorical encodings (adjust if your training used different mapping)
MAINT_MAP = {"None": 0, "Preventive": 1, "Corrective": 2}

# -------------------------------
# Load models (with friendly errors)
# -------------------------------
col_load1, col_load2 = st.columns(2)
with col_load1:
    try:
        clf = load_pickle("failure_multiclass.pkl")
        st.success("✅ Loaded multiclass failure model")
    except Exception as e:
        st.error(f"Couldn't load **failure_multiclass.pkl**. {e}")

with col_load2:
    try:
        reg = load_pickle("rul_regressor.pkl")
        st.success("✅ Loaded RUL regressor")
    except Exception as e:
        st.error(f"Couldn't load **rul_regressor.pkl**. {e}")

# Try to infer model feature names if user didn't save feature lists
clf_feature_names = getattr(getattr(clf, "get_booster", lambda: None)(), "feature_names", None)
reg_feature_names = getattr(getattr(reg, "get_booster", lambda: None)(), "feature_names", None)

# Load feature lists (preferred) or fall back to model-native names
clf_features = try_load_feature_list("clf_features.pkl", clf_feature_names)
reg_features = try_load_feature_list("rul_features.pkl", reg_feature_names)

if clf_features is None or reg_features is None:
    st.warning(
        "Feature lists were not found and could not be inferred from the models.\n\n"
        "Please save your feature lists as **clf_features.pkl** and **rul_features.pkl** "
        "to guarantee correct column order."
    )

# -------------------------------
# Inputs (union of all features needed by both models)
# -------------------------------
# If you know your full master set, define it here (order doesn’t matter; we reorder later).
master_features = list(dict.fromkeys((clf_features or []) + (reg_features or [])))

# Provide sensible UI defaults/ranges. Adjust to your data.
RANGES = {
    "SoC": (0, 100, 50),
    "SoH": (0, 100, 85),
    "Battery_Voltage": (100, 500, 350),
    "Battery_Current": (-200, 200, 0),
    "Battery_Temperature": (-20, 100, 30),
    "Charge_Cycles": (0, 5000, 500),
    "Motor_Temperature": (-10, 160, 60),
    "Motor_Vibration": (0.0, 50.0, 2.0),
    "Motor_Torque": (0, 600, 200),
    "Motor_RPM": (0, 20000, 3000),
    "Power_Consumption": (0, 500, 80),
    "Brake_Pad_Wear": (0, 100, 20),
    "Brake_Pressure": (0, 250, 60),
    "Reg_Brake_Efficiency": (0, 100, 75),
    "Tire_Pressure": (15, 55, 33),
    "Tire_Temperature": (-10, 120, 35),
    "Suspension_Load": (0, 3000, 300),
    "Ambient_Temperature": (-30, 55, 25),
    "Ambient_Humidity": (0, 100, 50),
    "Load_Weight": (0, 10000, 800),
    "Driving_Speed": (0, 220, 60),
    "Distance_Traveled": (0, 1_000_000, 50_000),
    "Idle_Time": (0, 1000, 10),
    "Route_Roughness": (0, 10, 3),
    # IMPORTANT: For the RUL regressor, RUL is the *target* (do NOT include as input).
    # If your classifier used RUL as a feature during training, keep it here; otherwise omit.
    "RUL": (0, 20000, 2000),
    # Maintenance_Type is categorical (encoded below)
    "TTF": (0, 20000, 1000),
    "Component_Health_Score": (0, 100, 90),
}

st.sidebar.header("🔧 Enter / Select Sensor Readings")

user_inputs = {}
# If we don't know the master features (e.g., no feature lists), offer a common sensible set
if not master_features:
    master_features = [
        'SoC','SoH','Battery_Voltage','Battery_Current','Battery_Temperature',
        'Charge_Cycles','Motor_Temperature','Motor_Vibration','Motor_Torque',
        'Motor_RPM','Power_Consumption','Brake_Pad_Wear','Brake_Pressure',
        'Reg_Brake_Efficiency','Tire_Pressure','Tire_Temperature','Suspension_Load',
        'Ambient_Temperature','Ambient_Humidity','Load_Weight','Driving_Speed',
        'Distance_Traveled','Idle_Time','Route_Roughness','RUL','TTF','Component_Health_Score',
        'Maintenance_Type'  # keep last for clarity
    ]

# Build UI controls
for feat in master_features:
    if feat == "Maintenance_Type":
        user_inputs[feat] = MAINT_MAP[
            st.sidebar.selectbox("Maintenance Type", list(MAINT_MAP.keys()), index=0)
        ]
    else:
        lo, hi, default = RANGES.get(feat, (0.0, 1000.0, 0.0))
        # choose slider type (int vs float) based on defaults
        if isinstance(lo, int) and isinstance(hi, int) and isinstance(default, int):
            user_inputs[feat] = st.sidebar.slider(feat, lo, hi, default)
        else:
            user_inputs[feat] = st.sidebar.slider(feat, float(lo), float(hi), float(default))

st.sidebar.markdown("---")
go = st.sidebar.button("🚀 Run Prediction")

# -------------------------------
# Layout
# -------------------------------
tab_pred, tab_explain = st.tabs(["🔮 Predictions", "🧠 Explainability"])

# -------------------------------
# Predictions
# -------------------------------
with tab_pred:
    st.subheader("📊 Model Outputs")

    if go:
        # Build inputs for each model in exactly the order they expect
        # (If feature lists are missing, we'll try to use the union order shown above.)
        if clf is not None:
            clf_feats = clf_features or master_features
            X_clf = ensure_dataframe_order(user_inputs, clf_feats)
        else:
            X_clf = None

        if reg is not None:
            reg_feats = [f for f in (reg_features or master_features) if f != "RUL"]
            X_reg = ensure_dataframe_order(user_inputs, reg_feats)
        else:
            X_reg = None

        # ---- Multi-class prediction
        if X_clf is not None:
            try:
                y_proba = None
                if hasattr(clf, "predict_proba"):
                    y_proba = clf.predict_proba(X_clf)[0]
                    pred_class = int(np.argmax(y_proba))
                else:
                    pred_class = int(clf.predict(X_clf)[0])

                label = CLASS_LABELS.get(pred_class, f"Class {pred_class}")
                if pred_class == 0:
                    st.success(f"Multiclass Prediction: **{label}**")
                else:
                    st.error(f"Multiclass Prediction: **{label}**")

                # Health score = probability of Healthy (class 0), scaled to 0–100
                if y_proba is not None and len(y_proba) >= 1:
                    p_healthy = float(y_proba[0])
                    health_score = max(0, min(100, round(p_healthy * 100, 1)))
                else:
                    # fallback (no proba): binary-ish score
                    health_score = 100.0 if pred_class == 0 else 20.0

                st.markdown("### ❤️ Health Score")
                st.progress(int(health_score))
                st.markdown(f"**{health_score}/100**")

                # Show class probabilities if available
                if y_proba is not None:
                    st.markdown("### 🧪 Class Probabilities")
                    probs_df = pd.DataFrame(
                        [y_proba],
                        columns=[CLASS_LABELS.get(i, f"Class {i}") for i in range(len(y_proba))]
                    ).T.rename(columns={0: "Probability"})
                    probs_df["Probability"] = probs_df["Probability"].round(3)
                    st.dataframe(probs_df)

            except Exception as e:
                st.error(f"Classification failed: {e}")

        # ---- RUL prediction
        if X_reg is not None:
            try:
                rul_pred = float(reg.predict(X_reg)[0])
                st.markdown("### ⏳ Remaining Useful Life (RUL)")
                st.info(f"Estimated RUL: **{rul_pred:,.2f}** (cycles / hours / km)")
            except Exception as e:
                st.error(f"RUL regression failed: {e}")

# -------------------------------
# Explainability
# -------------------------------
with tab_explain:
    st.subheader("🧠 Feature Importance (Top 10)")

    # Choose which model to explain
    explain_target = st.selectbox("Explain which model?", ["Multiclass Classifier", "RUL Regressor"])

    try:
        if explain_target == "Multiclass Classifier" and clf is not None:
            feats = clf_features or master_features
            if hasattr(clf, "feature_importances_"):
                importances = clf.feature_importances_
                imp_df = pd.DataFrame({"feature": feats, "importance": importances})
                imp_df = imp_df.sort_values("importance", ascending=False).head(10)
                st.bar_chart(imp_df.set_index("feature"))
            else:
                st.info("This classifier does not expose feature_importances_. Consider SHAP for deeper explainability.")
        elif explain_target == "RUL Regressor" and reg is not None:
            feats = reg_features or [f for f in master_features if f != "RUL"]
            if hasattr(reg, "feature_importances_"):
                importances = reg.feature_importances_
                imp_df = pd.DataFrame({"feature": feats, "importance": importances})
                imp_df = imp_df.sort_values("importance", ascending=False).head(10)
                st.bar_chart(imp_df.set_index("feature"))
            else:
                st.info("This regressor does not expose feature_importances_. Consider SHAP for deeper explainability.")
    except Exception as e:
        st.error(f"Could not render feature importance: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("🔋 Built with Streamlit • XGBoost/Sklearn • Joblib")
