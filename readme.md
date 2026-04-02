# 🧪 Shape Memory Alloy (SMA) Informatics Platform

An end-to-end Machine Learning web application designed to predict the thermodynamic transformation temperatures ($A_f, A_s, M_f, M_s$) of Shape Memory Alloys based on their elemental composition and processing conditions.

## 🚀 Features
* **Multi-Output Machine Learning:** Utilizes a Random Forest Multi-Output Regressor to predict highly coupled thermodynamic variables simultaneously.
* **Physics Engine:** Enforces thermodynamic laws (e.g., $A_s < A_f$) to prevent physically impossible predictions.
* **Smart Routing:** Automatically routes user input to the correct base-matrix model (e.g., Ni-Ti family vs. Cu-Al family).
* **Uncertainty Estimation:** Calculates prediction confidence intervals using Random Forest tree-variance estimation.
* **Modern Web Stack:** High-performance REST API built with **FastAPI** and an interactive, data-rich frontend built with **Streamlit**.

## 📁 Project Structure
```text
sma_platform/
├── api/
│   ├── main.py              # FastAPI server and endpoints
│   └── schemas.py           # Pydantic data validation models
├── core/
│   ├── physics.py           # Thermodynamic constraint logic
│   └── router.py            # Alloy family routing logic
├── data/                    # CSV datasets (e.g., niti_dummy_data.csv)
├── frontend/
│   └── app.py               # Streamlit interactive UI
├── models/
│   ├── pipeline.py          # ML training and prediction scripts
│   └── registry/            # Saved .pkl model files
├── train_initial.py         # Script to generate data and train the first model
└── README.md                # Project documentation

🛠️ Installation & Setup
1. Create and activate a virtual environment:

Bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
2. Install dependencies:

Bash
pip install fastapi uvicorn scikit-learn pandas numpy pydantic joblib streamlit requests matplotlib seaborn
3. Train the initial model:

Bash
python train_initial.py
🏃‍♂️ Running the Application
You need two terminal windows running simultaneously (both with the virtual environment activated).

Terminal 1: Start the Backend (FastAPI)

Bash
uvicorn api.main:app --reload
API Documentation available at: http://127.0.0.1:8000/docs

Terminal 2: Start the Frontend (Streamlit)

Bash
cd frontend
streamlit run app.py
Interactive UI available at: http://localhost:8501


---

### 2. The `ARCHITECTURE_EXPLANATION.md` File
Save this as `ARCHITECTURE_EXPLANATION.md` (or just keep it as notes for yourself). This explains exactly how the data flows from the user to the model and back.

```markdown
# 🧠 System Architecture & Workflow Explanation

This project is built using a modern **micro-service-like architecture**, separating the User Interface (Frontend), the Application Logic (Backend), and the Math (Machine Learning).

Here is the exact step-by-step lifecycle of a single prediction:

### Phase 1: The Frontend (Streamlit)
1. **User Input:** The user adjusts sliders in the Streamlit UI (`frontend/app.py`).
2. **Payload Creation:** Streamlit bundles these 22 numbers (19 elements, density, heating/cooling rates) into a standard JSON dictionary.
3. **HTTP Request:** Streamlit sends a `POST` request over the local network to the FastAPI backend URL (`http://127.0.0.1:8000/predict`).

### Phase 2: The Backend (FastAPI & Core)
4. **Validation (`api/schemas.py`):** FastAPI receives the JSON. It uses **Pydantic** to verify the math. *Did the user submit a negative Cooling Rate? Did they type a letter instead of a number?* If the data is bad, it instantly rejects it.
5. **Model Routing (`core/router.py`):** The router looks at the composition. If `Ni + Ti >= 50`, it decides this is a Ni-Ti alloy and tells the system to load the `NiTi_Family.pkl` model.
6. **Model Loading:** The system checks if the model is already in the server's RAM. If not, it loads it from `models/registry/`.

### Phase 3: Machine Learning (`models/pipeline.py`)
7. **Preprocessing:** The raw numbers are passed through a `StandardScaler` to normalize the data (making large numbers smaller so the ML algorithm doesn't get confused).
8. **Prediction:** The data passes through the `MultiOutputRegressor` (Random Forest). It predicts four highly correlated targets simultaneously: $A_f, A_s, M_f, M_s$.
9. **Uncertainty Calculation:** The system looks at all 100 decision trees inside the Random Forest. It calculates the *Standard Deviation* of their answers. If all trees agree, confidence is high (low variance). If the trees disagree wildly, confidence is low (meaning this is a highly unusual alloy composition).

### Phase 4: Physics Engine (`core/physics.py`)
10. **Thermodynamic Enforcement:** Machine learning algorithms only know math, not physics. It might accidentally predict that a material finishes transforming ($A_f$) *before* it starts ($A_s$). The Physics Engine steps in and corrects these physical impossibilities.
11. **TSPAN Calculation:** It calculates the Thermal Transformation Span ($A_f - M_f$). This is done *after* the ML prediction to prevent data leakage during training.

### Phase 5: The Return Journey
12. **API Response:** FastAPI bundles the final corrected predictions, the confidence intervals, and the routed model name into a new JSON file and sends it back to Streamlit.
13. **UI Rendering:** Streamlit unpacks the JSON and updates the colorful metric cards on the screen for the user to read!