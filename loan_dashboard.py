
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

df = pd.read_csv("loan_data.csv")


le_approval = LabelEncoder()
df['Approval'] = le_approval.fit_transform(df['Approval'])  

le_employment = LabelEncoder()
df['Employment_Status'] = le_employment.fit_transform(df['Employment_Status'])


df.drop(columns=['Text'], inplace=True)


X = df.drop(columns=['Approval'])
y = df['Approval']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


print("Classification Report:")
print(classification_report(y_test, model.predict(X_test)))

print("Confusion Matrix:")
print(confusion_matrix(y_test, model.predict(X_test)))

explainer = ClassifierExplainer(
    model, X_test, y_test,
    labels=["Rejected", "Approved"],
    descriptions={
        "Income": "Monthly Income of the applicant",
        "Credit_Score": "Credit score (e.g., 650, 700)",
        "Loan_Amount": "Loan amount requested",
        "DTI_Ratio": "Debt-to-Income Ratio",
        "Employment_Status": "Employment Status (encoded)"
    }
)


ExplainerDashboard(explainer,
                   title="Loan Approval Prediction Dashboard",
                   whatif=True,  # Enables What-If analysis
                   shap_interaction=False).run()
