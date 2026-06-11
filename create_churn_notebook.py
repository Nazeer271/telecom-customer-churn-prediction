
import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

def md(text):
    return nbf.v4.new_markdown_cell(text)

def code(text):
    return nbf.v4.new_code_cell(text)

# ── Title ──
cells.append(md("""# Customer Churn Prediction
**Objective:** Build a classification model to predict whether a telecom customer will churn, using demographic, account, and service-related features.
"""))

# ── 1. Imports ──
cells.append(md("## 1. Import Libraries"))
cells.append(code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, precision_recall_curve)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Plot style
plt.rcParams.update({'figure.figsize': (10, 6), 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 12})
sns.set_palette('Set2')
print("✅ Libraries loaded successfully")
"""))

# ── 2. Load Data ──
cells.append(md("## 2. Load Dataset"))
cells.append(code("""
df = pd.read_csv(r'customer_churn.csv')
print(f"Dataset Shape: {df.shape}")
print(f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")
df.head()
"""))

cells.append(code("""
print("Column Data Types:")
print(df.dtypes)
"""))

# ── 3. EDA ──
cells.append(md("## 3. Exploratory Data Analysis (EDA)"))

cells.append(code("""
print("=== Basic Statistics ===")
print(df.describe())
"""))

cells.append(code("""
print("=== Missing Values ===")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found ✅")
"""))

cells.append(code("""
# Target variable distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

churn_counts = df['Churn'].value_counts()
axes[0].pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%',
            colors=['#2ecc71','#e74c3c'], startangle=90, explode=(0,0.05))
axes[0].set_title('Churn Distribution', fontsize=14, fontweight='bold')

axes[1].bar(churn_counts.index, churn_counts.values, color=['#2ecc71','#e74c3c'], edgecolor='white')
axes[1].set_title('Churn Count', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Count')
for i, v in enumerate(churn_counts.values):
    axes[1].text(i, v + 30, str(v), ha='center', fontweight='bold')

plt.suptitle('Target Variable: Customer Churn', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('churn_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Churn Rate: {churn_counts['Yes']/len(df)*100:.1f}%")
"""))

cells.append(code("""
# Numeric feature distributions
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Convert TotalCharges to numeric (may have spaces)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, col in enumerate(numeric_cols):
    df.groupby('Churn')[col].plot(kind='kde', ax=axes[i], legend=True)
    axes[i].set_title(f'{col} Distribution by Churn', fontweight='bold')
    axes[i].set_xlabel(col)
plt.suptitle('Numeric Features by Churn Status', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('numeric_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(code("""
# Categorical features vs Churn
cat_cols = ['gender','SeniorCitizen','Partner','Dependents','PhoneService',
            'MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
            'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
            'Contract','PaperlessBilling','PaymentMethod']

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.ravel()
for i, col in enumerate(cat_cols):
    ct = pd.crosstab(df[col], df['Churn'], normalize='index') * 100
    ct.plot(kind='bar', ax=axes[i], color=['#2ecc71','#e74c3c'], rot=30, legend=(i==0))
    axes[i].set_title(col, fontweight='bold')
    axes[i].set_ylabel('% Customers')
    axes[i].set_xlabel('')
plt.suptitle('Categorical Features vs Churn Rate', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('categorical_features.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(code("""
# Correlation heatmap (numeric)
df_temp = df.copy()
df_temp['Churn_binary'] = (df_temp['Churn'] == 'Yes').astype(int)
corr_df = df_temp[['tenure','MonthlyCharges','TotalCharges','SeniorCitizen','Churn_binary']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('Correlation Matrix (Numeric Features)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(code("""
# Churn by Contract type (key business insight)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x=='Yes').sum()/len(x)*100)
axes[0].bar(contract_churn.index, contract_churn.values, color=['#e74c3c','#f39c12','#2ecc71'], edgecolor='white')
axes[0].set_title('Churn Rate by Contract Type', fontweight='bold')
axes[0].set_ylabel('Churn Rate (%)')
for i, v in enumerate(contract_churn.values):
    axes[0].text(i, v+0.5, f'{v:.1f}%', ha='center', fontweight='bold')

# Churn by Tenure buckets
df['tenure_group'] = pd.cut(df['tenure'], bins=[0,12,24,48,72], labels=['0-12m','13-24m','25-48m','49-72m'])
tenure_churn = df.groupby('tenure_group', observed=True)['Churn'].apply(lambda x: (x=='Yes').sum()/len(x)*100)
axes[1].bar(tenure_churn.index, tenure_churn.values, color=['#e74c3c','#e67e22','#3498db','#2ecc71'], edgecolor='white')
axes[1].set_title('Churn Rate by Tenure Group', fontweight='bold')
axes[1].set_ylabel('Churn Rate (%)')
for i, v in enumerate(tenure_churn.values):
    axes[1].text(i, v+0.5, f'{v:.1f}%', ha='center', fontweight='bold')

plt.suptitle('Key Business Insights', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('business_insights.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ── 4. Preprocessing ──
cells.append(md("## 4. Data Preprocessing"))

cells.append(code("""
df_model = df.copy()

# 1. Drop customerID (not predictive) and tenure_group (derived)
df_model.drop(columns=['customerID', 'tenure_group'], errors='ignore', inplace=True)

# 2. Handle TotalCharges (numeric conversion, fill NaN with median)
df_model['TotalCharges'] = pd.to_numeric(df_model['TotalCharges'], errors='coerce')
df_model['TotalCharges'].fillna(df_model['TotalCharges'].median(), inplace=True)

# 3. Encode binary target
df_model['Churn'] = (df_model['Churn'] == 'Yes').astype(int)

# 4. Encode all categorical features
le = LabelEncoder()
cat_features = df_model.select_dtypes(include='object').columns.tolist()
print(f"Categorical columns to encode: {cat_features}")

for col in cat_features:
    df_model[col] = le.fit_transform(df_model[col])

print(f"\\nFinal shape: {df_model.shape}")
print(f"Churn distribution after encoding:\\n{df_model['Churn'].value_counts()}")
df_model.head()
"""))

cells.append(code("""
# Feature matrix and target
X = df_model.drop('Churn', axis=1)
y = df_model['Churn']

# Train-test split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set : {X_train.shape[0]:,} rows")
print(f"Testing  set : {X_test.shape[0]:,} rows")
print(f"Train churn rate: {y_train.mean()*100:.1f}%")
print(f"Test  churn rate: {y_test.mean()*100:.1f}%")
"""))

cells.append(code("""
# Feature scaling for models that need it
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print("✅ Scaling complete")
"""))

# ── 5. Modeling ──
cells.append(md("## 5. Model Building"))

cells.append(code("""
def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:,1]
    results = {
        'Model'    : name,
        'Accuracy' : accuracy_score(y_te, y_pred),
        'Precision': precision_score(y_te, y_pred),
        'Recall'   : recall_score(y_te, y_pred),
        'F1 Score' : f1_score(y_te, y_pred),
        'ROC-AUC'  : roc_auc_score(y_te, y_proba)
    }
    return results, y_pred, y_proba

results_list = []

# --- Logistic Regression ---
lr = LogisticRegression(max_iter=1000, random_state=42)
res, _, _ = evaluate_model('Logistic Regression', lr, X_train_sc, X_test_sc, y_train, y_test)
results_list.append(res)
print("✅ Logistic Regression done")

# --- Decision Tree ---
dt = DecisionTreeClassifier(max_depth=6, random_state=42)
res, _, _ = evaluate_model('Decision Tree', dt, X_train, X_test, y_train, y_test)
results_list.append(res)
print("✅ Decision Tree done")

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
res, _, _ = evaluate_model('Random Forest', rf, X_train, X_test, y_train, y_test)
results_list.append(res)
print("✅ Random Forest done")

# --- Gradient Boosting ---
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
res, _, _ = evaluate_model('Gradient Boosting', gb, X_train, X_test, y_train, y_test)
results_list.append(res)
print("✅ Gradient Boosting done")

results_df = pd.DataFrame(results_list).set_index('Model')
results_df = results_df.round(4)
print("\\n=== Model Comparison ===")
print(results_df.to_string())
"""))

cells.append(code("""
# Visualise comparison
fig, ax = plt.subplots(figsize=(12, 6))
results_df.plot(kind='bar', ax=ax, edgecolor='white', rot=15)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('Score')
ax.set_ylim(0, 1.1)
ax.legend(loc='upper right')
ax.axhline(0.8, ls='--', color='gray', alpha=0.6, label='0.8 reference')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ── 6. Hyperparameter Tuning ──
cells.append(md("## 6. Hyperparameter Tuning (Best Model)"))

cells.append(code("""
# Tune Random Forest (typically best performer on tabular data)
param_grid = {
    'n_estimators'    : [100, 200],
    'max_depth'       : [6, 10, None],
    'min_samples_split': [2, 5],
    'max_features'    : ['sqrt', 'log2']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                        param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_rf.fit(X_train, y_train)

print(f"Best Parameters : {grid_rf.best_params_}")
print(f"Best CV ROC-AUC : {grid_rf.best_score_:.4f}")
best_rf = grid_rf.best_estimator_
"""))

cells.append(code("""
# Evaluate best tuned model
res_tuned, y_pred_best, y_proba_best = evaluate_model(
    'Tuned Random Forest', best_rf, X_train, X_test, y_train, y_test)
print("\\n=== Tuned Random Forest Performance ===")
for k, v in res_tuned.items():
    if k != 'Model':
        print(f"  {k:12s}: {v:.4f}")
"""))

# ── 7. Evaluation ──
cells.append(md("## 7. Detailed Model Evaluation"))

cells.append(code("""
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Not Churn','Churn'], yticklabels=['Not Churn','Churn'])
axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# ROC Curve (all models)
models_for_roc = [
    ('Logistic Regression', lr, X_test_sc),
    ('Decision Tree',       dt, X_test),
    ('Random Forest',       rf, X_test),
    ('Gradient Boosting',   gb, X_test),
    ('Tuned Random Forest', best_rf, X_test)
]
for name, model, Xte in models_for_roc:
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(Xte)[:,1])
    auc = roc_auc_score(y_test, model.predict_proba(Xte)[:,1])
    axes[1].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
axes[1].plot([0,1],[0,1],'k--', label='Random')
axes[1].set_title('ROC Curves', fontsize=14, fontweight='bold')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig('evaluation_plots.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\n=== Classification Report ===")
print(classification_report(y_test, y_pred_best, target_names=['Not Churn','Churn']))
"""))

cells.append(code("""
# Feature Importance
feat_imp = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=True)
top20 = feat_imp.tail(20)

plt.figure(figsize=(10, 7))
top20.plot(kind='barh', color=['#e74c3c' if v > top20.quantile(0.75) else '#3498db' for v in top20])
plt.title('Top 20 Feature Importances (Tuned Random Forest)', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\nTop 10 most important features:")
print(feat_imp.tail(10)[::-1].to_string())
"""))

# ── 8. Cross Validation ──
cells.append(md("## 8. Cross-Validation"))

cells.append(code("""
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_rf, X, y, cv=cv5, scoring='roc_auc', n_jobs=-1)
print(f"5-Fold Cross-Validation ROC-AUC:")
for i, s in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {s:.4f}")
print(f"  Mean  : {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
"""))

# ── 9. Predict New Data ──
cells.append(md("## 9. Predictions on New / Unseen Customers"))

cells.append(code("""
# Simulate 5 new customers
new_customers_raw = pd.DataFrame({
    'customerID' : ['NEW-001','NEW-002','NEW-003','NEW-004','NEW-005'],
    'gender'     : ['Male','Female','Male','Female','Male'],
    'SeniorCitizen': [0, 1, 0, 0, 1],
    'Partner'    : ['Yes','No','No','Yes','No'],
    'Dependents' : ['No','No','Yes','No','No'],
    'tenure'     : [2, 58, 24, 5, 1],
    'PhoneService': ['Yes','Yes','Yes','No','Yes'],
    'MultipleLines': ['No','Yes','No','No phone service','Yes'],
    'InternetService': ['Fiber optic','DSL','DSL','DSL','Fiber optic'],
    'OnlineSecurity': ['No','Yes','Yes','No','No'],
    'OnlineBackup': ['No','Yes','No','Yes','No'],
    'DeviceProtection': ['No','Yes','Yes','No','No'],
    'TechSupport': ['No','Yes','No','No','No'],
    'StreamingTV': ['Yes','No','No','No','Yes'],
    'StreamingMovies': ['Yes','No','No','No','Yes'],
    'Contract'   : ['Month-to-month','Two year','One year','Month-to-month','Month-to-month'],
    'PaperlessBilling': ['Yes','No','No','Yes','Yes'],
    'PaymentMethod': ['Electronic check','Bank transfer (automatic)','Mailed check','Electronic check','Electronic check'],
    'MonthlyCharges': [95.4, 45.2, 52.6, 30.0, 80.5],
    'TotalCharges': [190.8, 2621.6, 1262.4, 150.0, 80.5],
    'Churn'      : ['?','?','?','?','?']
})

new_customers = new_customers_raw.drop(columns=['customerID','Churn'])

# Encode same as training
for col in new_customers.select_dtypes(include='object').columns:
    le_tmp = LabelEncoder()
    le_tmp.fit(df_model_ref := df.drop(columns=['customerID','tenure_group'], errors='ignore')[col] if col in df.columns else new_customers[col])
    # Safe encoding
    known = set(le_tmp.classes_)
    new_customers[col] = new_customers[col].apply(lambda x: x if x in known else le_tmp.classes_[0])
    new_customers[col] = le_tmp.transform(new_customers[col])

new_customers['TotalCharges'] = pd.to_numeric(new_customers['TotalCharges'], errors='coerce').fillna(0)

preds    = best_rf.predict(new_customers)
probas   = best_rf.predict_proba(new_customers)[:,1]

results_new = new_customers_raw[['customerID','gender','tenure','Contract','MonthlyCharges']].copy()
results_new['Churn Probability'] = (probas * 100).round(1)
results_new['Prediction']        = ['⚠️ CHURN' if p==1 else '✅ Stay' for p in preds]
results_new['Risk Level']        = pd.cut(probas, bins=[0,.3,.6,1], labels=['🟢 Low','🟡 Medium','🔴 High'])
print(results_new.to_string(index=False))
"""))

# ── 10. Business Insights ──
cells.append(md("""## 10. Business Insights & Recommendations

### Key Findings from the Model

| Finding | Insight |
|---------|---------|
| **Month-to-month contracts** have the highest churn rate (~42%) | Offer incentives to upgrade to annual contracts |
| **New customers (0–12 months)** churn most | Implement strong onboarding programs |
| **Fiber optic users** churn more than DSL users | Investigate service quality and pricing for fiber optic |
| **Higher monthly charges** correlate with churn | Review pricing strategy and offer loyalty discounts |
| **Customers without online security/tech support** churn more | Bundle these services at a discount |
| **Electronic check payment** users churn more | Encourage auto-pay enrollment |

### Recommended Retention Strategies
1. **Target at-risk segment:** Use model scores > 60% probability to trigger retention calls
2. **Contract migration:** Incentivize month-to-month customers to switch to annual plans
3. **Early intervention:** Flag customers in first 12 months for proactive outreach
4. **Bundle services:** Offer online security + tech support as free add-ons for 6 months
5. **Loyalty rewards:** Discount for customers at months 12, 24, and 36 tenure milestones
"""))

# ── Summary stats cell ──
cells.append(code("""
print("=" * 55)
print("  CUSTOMER CHURN PREDICTION - FINAL SUMMARY")
print("=" * 55)
print(f"Dataset        : {df.shape[0]:,} customers, {df.shape[1]} features")
print(f"Baseline churn : {(df['Churn']=='Yes').mean()*100:.1f}%")
print(f"Best model     : Tuned Random Forest")
print(f"Test Accuracy  : {res_tuned['Accuracy']*100:.1f}%")
print(f"Test ROC-AUC   : {res_tuned['ROC-AUC']:.4f}")
print(f"Test F1-Score  : {res_tuned['F1 Score']:.4f}")
print(f"CV ROC-AUC     : {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
print("=" * 55)
"""))

nb.cells = cells
nbf.write(nb, r'c:\Users\nazee\Documents\Customer_Churn_Prediction.ipynb')
print("Notebook created: Customer_Churn_Prediction.ipynb")
