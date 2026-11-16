"""
Heart attack risk classifier (ANN)
Dataset : Personal Key Indicators of Heart Disease (CDC BRFSS 2022)
          ~246,000 rows, ~39 features, target = HadHeartAttack (Yes/No)
Get data: Kaggle -> kamilpytlak/personal-key-indicators-of-heart-disease
          download heart_2022_no_nans.csv and put it next to this file
          (or set up Kaggle credentials and the script downloads it itself)
Run     : pip install -r requirements.txt && python train.py
Outputs : model.keras, model_weights.js, confusion_matrix.png,
          loss_accuracy.png, classification_report.txt
To use another CSV, change CSV_NAME, TARGET, POSITIVE and DROP below.
"""
import os, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

CSV_NAME = "heart_2022_no_nans.csv"
TARGET, POSITIVE = "HadHeartAttack", "Yes"
DROP = ["State"]
SEED = 42
tf.random.set_seed(SEED); np.random.seed(SEED)

# 1. Dataset ----------------------------------------------------------------
if not os.path.exists(CSV_NAME):
    try:
        import kagglehub
        path = kagglehub.dataset_download("kamilpytlak/personal-key-indicators-of-heart-disease")
        found = glob.glob(os.path.join(path, "**", CSV_NAME), recursive=True)
        df = pd.read_csv(found[0])
    except Exception as e:
        raise SystemExit(f"Put {CSV_NAME} next to train.py (download from Kaggle). Details: {e}")
else:
    df = pd.read_csv(CSV_NAME)

df = df.drop(columns=[c for c in DROP if c in df.columns]).dropna()
y = (df[TARGET] == POSITIVE).astype(int).values
X = df.drop(columns=[TARGET])
print("Shape:", X.shape, "| Positive rate:", y.mean().round(3))

# Build schema (drives both training and the website form)
schema, cols = [], []
for c in X.columns:
    if pd.api.types.is_numeric_dtype(X[c]):
        schema.append({"name": c, "type": "num", "min": float(X[c].min()),
                       "max": float(X[c].max()), "default": float(X[c].median())})
        cols.append(X[c].astype(float).values)
    else:
        opts = sorted(X[c].astype(str).unique().tolist())
        schema.append({"name": c, "type": "cat", "options": opts,
                       "default": X[c].astype(str).mode()[0]})
        for o in opts:
            cols.append((X[c].astype(str) == o).astype(float).values)
M = np.column_stack(cols)
print("Encoded feature count:", M.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    M, y, test_size=0.2, stratify=y, random_state=SEED)
scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
class_weight = {0: cw[0], 1: cw[1]}

# 2. ANN model ----------------------------------------------------------------
model = keras.Sequential([
    layers.Input(shape=(M.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(
    X_train_s, y_train, validation_split=0.15, epochs=40, batch_size=512,
    class_weight=class_weight,
    callbacks=[keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)],
    verbose=2)

# 3. Confusion matrix + classification report ---------------------------------
y_pred = (model.predict(X_test_s, batch_size=2048).ravel() >= 0.5).astype(int)
labels = ["No heart attack", "Heart attack"]
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4.8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion matrix")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.close()

report = classification_report(y_test, y_pred, target_names=labels)
print(report); open("classification_report.txt", "w").write(report)

# 4. Loss vs accuracy ----------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, k, t in zip(ax, ["loss", "accuracy"], ["Loss", "Accuracy"]):
    a.plot(history.history[k], label="train")
    a.plot(history.history["val_" + k], label="validation")
    a.set_title(t); a.set_xlabel("Epoch"); a.legend()
plt.tight_layout(); plt.savefig("loss_accuracy.png", dpi=150); plt.close()

# 5. Store model + export for the website ---------------------------------------
model.save("model.keras")
dense = [l for l in model.layers if isinstance(l, layers.Dense)]
export = {
    "schema": schema,
    "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
    "layers": [{"W": l.get_weights()[0].round(5).tolist(),
                "b": l.get_weights()[1].round(5).tolist(),
                "act": "sigmoid" if i == len(dense) - 1 else "relu"}
               for i, l in enumerate(dense)],
    "test_accuracy": float((y_pred == y_test).mean()),
    "rows": int(len(df)), "target": "heart attack",
}
open("model_weights.js", "w").write("window.MODEL = " + json.dumps(export) + ";")
print("Saved model.keras and model_weights.js")
