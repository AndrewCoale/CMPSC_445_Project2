import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_excel("Trump_Tariff_Data.xlsx")

# Feature engineering: create proxy targets
df["Trade Balance"] = df["US 2024 Exports"] - df["US 2024 Imports (Customs Basis)"]
df["Deficit per Capita"] = df["US 2024 Deficit"] / df["Population"]

# Drop rows with missing target values
df_clean = df.dropna(subset=["Trade Balance", "Deficit per Capita"]).copy()

# Define features and targets
features = ["Country", "Trump Tariffs Alleged", "US 2024 Exports", "US 2024 Imports (Customs Basis)", "Population"]
X = df_clean[features]
y_gdp = df_clean["Trade Balance"]
y_inflation = df_clean["Deficit per Capita"]

# Preprocessing
categorical = ["Country", "Trump Tariffs Alleged"]
numerical = ["US 2024 Exports", "US 2024 Imports (Customs Basis)", "Population"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), numerical)
    ]
)

# Pipelines
gdp_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

inflation_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

# Train and save models
gdp_model.fit(X, y_gdp)
inflation_model.fit(X, y_inflation)

joblib.dump(gdp_model, "gdp_model.pkl")
joblib.dump(inflation_model, "inflation_model.pkl")
