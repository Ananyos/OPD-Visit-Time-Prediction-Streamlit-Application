# 🏥 Hospital A — OPD Visit Time Prediction

A machine learning web app that predicts total outpatient department (OPD) visit time across 3 hospital branches, built with scikit-learn and Streamlit.

---

## 📁 Project Files

```
opd-app/
├── opd_mock_data.csv           ← Dataset (1,000 OPD visit records)
├── opd_prediction_model.py     ← ML model training script
├── streamlit_app.py            ← Streamlit web app
└── opd_model.pkl               ← Auto-generated after first training run
```

> ⚠️ All 3 files (`opd_mock_data.csv`, `opd_prediction_model.py`, `streamlit_app.py`) **must be in the same folder** before you begin.

---

## ✅ Requirements

| Tool | Minimum version | Check command |
|---|---|---|
| Python | 3.8+ | `python --version` |
| pip | any | `pip --version` |

---

## 🪟 Windows Setup

### Step 1 — Open the project folder in VS Code

1. Create a folder called `opd-app` anywhere on your computer (e.g. `C:\Users\YourName\opd-app`)
2. Place all 3 project files inside it
3. Open **VS Code** → `File` → `Open Folder` → select `opd-app`
4. Open the terminal: **Terminal** → **New Terminal** (or press `` Ctrl + ` ``)

---

### Step 2 — Create a virtual environment (recommended)

In the VS Code terminal, run these commands **one at a time**:

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

> You should see `(venv)` appear at the start of the terminal line. This means the environment is active.

---

### Step 3 — Install required packages

```bash
pip install pandas numpy scikit-learn streamlit
```

Wait for all packages to finish installing before moving on.

---

### Step 4 — Train the model

Run this **once** to train the ML model and save it as `opd_model.pkl`:

```bash
python opd_prediction_model.py
```

**Expected output:**
```
Dataset loaded successfully
  Rows      : 1,000
  ...
  MAE  : 9.19 mins
  RMSE : 11.16 mins
  ✅  Model saved to: opd_model.pkl
  ✅  All done. Ready for Streamlit integration.
```

> After this step, a new file `opd_model.pkl` will appear in your folder. You only need to run this step **once**.

---

### Step 5 — Launch the Streamlit app

```bash
streamlit run streamlit_app.py
```

Your browser will open automatically at:

```
http://localhost:8501
```

> To stop the app, press `Ctrl + C` in the terminal.

---

## 🍎 Mac Setup

### Step 1 — Open the project folder in VS Code

1. Create a folder called `opd-app` anywhere (e.g. `~/Desktop/opd-app`)
2. Place all 3 project files inside it
3. Open **VS Code** → `File` → `Open Folder` → select `opd-app`
4. Open the terminal: **Terminal** → **New Terminal** (or press `` Cmd + ` ``)

---

### Step 2 — Create a virtual environment (recommended)

In the VS Code terminal, run these commands **one at a time**:

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

> You should see `(venv)` appear at the start of the terminal line. This means the environment is active.

---

### Step 3 — Install required packages

```bash
pip install pandas numpy scikit-learn streamlit
```

Wait for all packages to finish installing before moving on.

---

### Step 4 — Train the model

Run this **once** to train the ML model and save it as `opd_model.pkl`:

```bash
python3 opd_prediction_model.py
```

**Expected output:**
```
Dataset loaded successfully
  Rows      : 1,000
  ...
  MAE  : 9.19 mins
  RMSE : 11.16 mins
  ✅  Model saved to: opd_model.pkl
  ✅  All done. Ready for Streamlit integration.
```

> After this step, a new file `opd_model.pkl` will appear in your folder. You only need to run this step **once**.

---

### Step 5 — Launch the Streamlit app

```bash
streamlit run streamlit_app.py
```

Your browser will open automatically at:

```
http://localhost:8501
```

> To stop the app, press `Ctrl + C` in the terminal.

---

## 🔁 Returning to the Project (After First Setup)

Every time you open VS Code again, just re-activate your virtual environment and launch the app:

**Windows:**
```bash
venv\Scripts\activate
streamlit run streamlit_app.py
```

**Mac:**
```bash
source venv/bin/activate
streamlit run streamlit_app.py
```

> No need to reinstall packages or retrain the model — `opd_model.pkl` is already saved.

---

## 🛠️ Troubleshooting

### `python` not recognised (Windows)
Try using `py` instead:
```bash
py -m venv venv
py opd_prediction_model.py
```

### `streamlit: command not found`
Run Streamlit as a Python module:
```bash
# Windows
python -m streamlit run streamlit_app.py

# Mac
python3 -m streamlit run streamlit_app.py
```

### `ModuleNotFoundError: No module named 'streamlit'`
Your virtual environment may not be active. Run the activate command again:
```bash
# Windows
venv\Scripts\activate

# Mac
source venv/bin/activate
```
Then re-run the install step.

### `FileNotFoundError: opd_mock_data.csv`
Make sure all 3 project files are in the **same folder** and that VS Code has that folder open (check the Explorer panel on the left).

### `FileNotFoundError: opd_model.pkl`
You need to train the model first. Run Step 4 again:
```bash
# Windows
python opd_prediction_model.py

# Mac
python3 opd_prediction_model.py
```

### Browser does not open automatically
Open your browser manually and go to:
```
http://localhost:8501
```

---

## 🧠 How the Model Works

| Feature | Description |
|---|---|
| `day_of_week` | Derived from the date (Mon–Sun, encoded 0–6) |
| `arrival_hour` | Start hour of the selected time slot |
| `service_type` | Label-encoded from 8 service categories |
| `branch` | Label-encoded (Branch 1, 2, 3) |
| `queue_size` | Simulated integer — patients ahead at arrival |

**Algorithm:** Random Forest Regressor (100 trees)  
**Train / Test split:** 80% / 20%  
**MAE:** ~9.2 minutes &nbsp;·&nbsp; **RMSE:** ~11.2 minutes

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
streamlit
```

Install all at once:
```bash
pip install pandas numpy scikit-learn streamlit
```

---

## 📌 Quick Reference

| Task | Windows | Mac |
|---|---|---|
| Activate environment | `venv\Scripts\activate` | `source venv/bin/activate` |
| Train model | `python opd_prediction_model.py` | `python3 opd_prediction_model.py` |
| Run app | `streamlit run streamlit_app.py` | `streamlit run streamlit_app.py` |
| Stop app | `Ctrl + C` | `Ctrl + C` |
| Open in browser | `http://localhost:8501` | `http://localhost:8501` |

---

*Hospital A · OPD Visit Time Prediction System · Prototype v1.0*
