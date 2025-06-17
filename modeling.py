import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("data/MachineLearningRating_v3.txt", delimiter="|")

# Step 1: Add computed fields
df["ClaimOccurred"] = (df["TotalClaims"] > 0).astype(int)
df["Margin"] = df["TotalPremium"] - df["TotalClaims"]

# Step 2: Drop rows with missing critical values
df = df.dropna(subset=["TotalPremium", "TotalClaims"])

# Step 3: Filter only rows with claims > 0 for claim severity prediction
df_claims = df[df["TotalClaims"] > 0].copy()

# Step 4: Encode categorical columns (label encoding for now)
cat_cols = df_claims.select_dtypes(include="object").columns
le = LabelEncoder()
for col in cat_cols:
    try:
        df_claims[col] = le.fit_transform(df_claims[col].astype(str))
    except:
        print(f"Skipping column: {col}")

# Step 5: Select features for modeling
features = df_claims.drop(columns=["TotalClaims", "UnderwrittenCoverID", "PolicyID"])
target = df_claims["TotalClaims"]

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

print("✅ Data ready for modeling.")



from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Helper function
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"\n{name} Results:")
    print(f"🔢 RMSE: {rmse:.2f}")
    print(f"📈 R²: {r2:.4f}")
    return model

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


# Linear Regression
lr = LinearRegression()
evaluate_model("Linear Regression", lr, X_train, X_test, y_train, y_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
evaluate_model("Random Forest", rf, X_train, X_test, y_train, y_test)

# XGBoost
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
evaluate_model("XGBoost", xgb, X_train, X_test, y_train, y_test)



import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Use trained random forest model
importances = rf.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(10)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df, x='Importance', y='Feature')
plt.title('Top 10 Important Features (Random Forest)')
plt.tight_layout()
plt.show()
