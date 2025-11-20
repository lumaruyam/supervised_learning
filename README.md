# 🐾 **Predicting Pet Adoption Likelihood**

### *Supervised Learning Final Project*

**Contributors:**

- **Julia Randriatsimivony**  
- **Luli Maruyama**

---

# 📌 **1. Business Use Case (BUC)**

## **Context**

Animal shelters often operate under severe resource constraints: limited staff, limited budget, and overcrowded facilities. Some animals are adopted quickly, while others stay in the shelter for weeks or months. Long stays increase operational costs and negatively impact animal well-being (stress, illness risk, behavioral deterioration).

---

## **Business Problem**

Shelters need a way to identify early which animals are less likely to be adopted, so they can allocate resources more effectively. For example:

* Boosting visibility for specific pets (special photos, social media posts)
* Providing additional medical or behavioral support
* Prioritizing foster placements
* Adjusting placement strategies before long shelter stays occur

---

## **Stakeholders**

* Shelter directors & operational managers
* Veterinarians & behavioral specialists
* Adoption coordinators
* Volunteers & foster networks

---

## **Goal**

Build a Machine Learning model that predicts the likelihood that an animal will be adopted, based on characteristics such as breed, age, health condition, size, and time in shelter.
This prediction helps shelters prioritize care and resources toward pets at high risk of prolonged stays.

---

## **Constraints**

* Limited dataset size
* Some noisy or subjective fields (e.g., HealthCondition, PreviousOwner)
* Imbalanced classes (more pets get adopted than not)

---

# 📊 **2. Dataset Description**

The dataset represents pets from a real-world adoption environment (non-synthetic), containing **2,007 rows** and **12 columns**.

Data Source: Kaggle (https://www.kaggle.com/datasets/rabieelkharoua/predict-pet-adoption-status-dataset)

---

### **Features**

| Feature           | Type         | Description                                     |
| ------------------| ------------ | ----------------------------------------------- |
| PetID             | Identifier   | Unique identifier for each pet                  |
| PetType           | Categorical  | Type of pet (e.g., Dog, Cat, Bird, Rabbit)      |
| Breed             | Categorical  | Specific breed of the pet                       |
| AgeMonths         | Numeric      | Age of the pet in months                        |
| Color             | Categorical  | Color of the pet                                |
| Size              | Categorical  | Size category (Small, Medium, Large)            |
| WeightKg          | Numeric      | Weight of the pet in kilograms                  |
| Vaccinated        | Binary (0/1) | 0 = Not vaccinated, 1 = Vaccinated              |
| HealthCondition   | Binary (0/1) | 0 = Healthy, 1 = Medical condition              |
| TimeInShelterDays | Numeric      | Number of days the pet has spent in the shelter |
| AdoptionFee       | Numeric      | Adoption fee charged (in USD)                   |
| PreviousOwner     | Binary (0/1) | 0 = No previous owner, 1 = Had previous owner   |

### **Target**

| Target Column            | Description                            |
| ------------------------ | -------------------------------------- |
| AdoptionLikelihood (0/1) | 1 = likely to be adopted, 0 = unlikely |

---

# 🔍 **3. Exploratory Data Analysis (EDA)**

The notebook `eda.ipynb` includes:

* Distribution of target classes
* Missing values and dtype analysis
* Histograms for numeric features
* Value counts for categorical features
* Correlation heatmaps
* Feature–target relationships (e.g., age vs adoption likelihood)
* Categorical patterns (e.g., breed adoption trends)

EDA conclusions guided our preprocessing and model selection.

---

# ⚙️ **4. Project Structure**

```
.
├── README.md
├── artifacts
│   ├── classification_report.json
│   └── confusion_matrix.json
├── eda.ipynb
├── main.ipynb
├── main.py
├── mlflow.db
```

---

# 🧪 **5. How to Reproduce the Project**

## **Environment Setup**

**Python version:** 3.11+

### **Install dependencies**

```bash
pip install -r requirements.txt
```

### **Dataset Placement**

Place your dataset file in:

```
./pets.csv
```

### **Run Training Pipeline**

```bash
python main.py
```

This will:

* Load dataset
* Run preprocessing
* Train the model
* Output metrics

---

# 🏗️ **6. Baseline Model**

We built a **DummyClassifier (most_frequent)** baseline.

### **Baseline Accuracy:**

`0.6816`

This establishes the minimum performance any ML model must exceed.

---

# 🔧 **7. Final Pipeline & Feature Engineering**

We developed a full preprocessing + model pipeline:

### **Preprocessing**

* OneHotEncoding for: `PetType`, `Breed`, `Color`
* OrdinalEncoding for: `Size (Small < Medium < Large)`
* StandardScaler for numeric features:

  * AgeMonths
  * WeightKg
  * TimeInShelterDays
  * AdoptionFee

---

### **Model**

We tested several models throughout development:

| Model                       | Result (Earlier Tests) |
| --------------------------- | ---------------------- |
| DummyClassifier             | 0.68 accuracy          |
| RandomForestClassifier      | ~0.98 accuracy         |
| GradientBoostingClassifier  | ~0.89 accuracy         |
| Random Forest (Grid Search) | ~0.888 accuracy        |

### **Best Performing Model (Early Phase)**

**RandomForestClassifier** using the full preprocessing pipeline produced the highest early accuracy (~0.98).

---

# **Final Cleaned Baseline (Latest Pipeline Results)**

After building the **final MLflow-tracked pipeline**, adding a stricter preprocessing workflow, and evaluating on the final hold-out test set, the realistic final performance is:

### **Final Evaluation Metrics (Updated & Cleaned)**

```
Accuracy: 0.9154
```

#### **Classification Report**

```
Class 0 → Precision 0.9275 | Recall 0.9481 | F1 0.9377
Class 1 → Precision 0.8889 | Recall 0.8484 | F1 0.8682
Macro F1 → 0.9030
Weighted F1 → 0.9149
```

#### **Confusion Matrix**

```
[[256, 14],
 [ 20,112]]
```

### **Interpretation**

* The earlier 0.98 accuracy came from preliminary experiments (different splits & no MLflow tracking).
* The final registered pipeline (with nested MLflow runs, proper preprocessing, and fixed random states) achieves a **stable and realistic ~0.915 accuracy**.
* This is the authoritative final metric.

---

# 🔬 **8. Experiment Tracking**

We recorded all experiments through MLflow and versioned commits:

| Experiment | Description                          | Result                   |
| ---------- | ------------------------------------ | ------------------------ |
| #0         | Dummy baseline                       | 0.681                    |
| #1         | RandomForest + minimal preprocessing | 0.95–0.98 (early tests)  |
| #2         | Full preprocessing pipeline          | ~0.98 (early tests)      |
| #3         | Hyperparameter search (RF)           | ~0.885–0.889             |
| #4         | Cross-validation                     | Stable around ~0.88–0.89 |
| #5         | GradientBoostingClassifier           | ~0.89                    |
| **#6**     | **Final MLflow pipeline (cleaned)**  | **0.915** (final)        |

---

### **Conclusion**

Although earlier prototypes reached ~0.98 accuracy, the **final validated pipeline** tracked with MLflow achieves **~0.915 accuracy**, which is:

* more realistic
* more stable
* reproducible
* fully integrated with preprocessing + nested MLflow tracking

---

# 📈 **9. Conclusion**

Our machine learning pipeline successfully predicts adoption likelihood with **very high accuracy**, enabling shelters to:

* Proactively support animals unlikely to be adopted
* Allocate medical and behavioral resources
* Improve adoption outcomes
* Reduce shelter overcrowding
* Enhance animal welfare

This project demonstrates the business value of data-driven decision making in the animal welfare sector.
