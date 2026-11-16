# Heart Attack Risk Classifier (ANN)

Dataset: Personal Key Indicators of Heart Disease (CDC BRFSS 2022), ~246k rows.
Kaggle: kamilpytlak/personal-key-indicators-of-heart-disease
Download `heart_2022_no_nans.csv` and place it beside train.py.

    pip install -r requirements.txt
    python train.py

Creates model.keras, model_weights.js, confusion_matrix.png,
loss_accuracy.png, classification_report.txt.

Website: open index.html (needs model_weights.js beside it).
The form is generated from the dataset, so any CSV works after editing
CSV_NAME / TARGET / POSITIVE / DROP in train.py.

## GitHub Pages
1. New repo, upload all files.
2. Settings -> Pages -> Deploy from branch -> main / root.
3. Live at https://<username>.github.io/<repo>/
