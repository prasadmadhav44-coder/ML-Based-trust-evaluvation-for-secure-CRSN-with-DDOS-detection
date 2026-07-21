# ============================================================
# ENHANCED: ML Benchmarking + Hyperparameter Tuning + Advanced Feature Importance
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

np.random.seed(42)

# ---------------- NETWORK SETUP ----------------
NUM_NODES = 80
AREA = 50
positions = np.random.rand(NUM_NODES, 2) * AREA

# ---------------- ATTACK TYPES ----------------
labels = np.random.choice(
    ["normal", "blackhole", "selective", "onoff", "ddos"],
    size=NUM_NODES,
    p=[0.5, 0.15, 0.15, 0.1, 0.1]
)

packets_sent = np.random.randint(80, 120, NUM_NODES)
packets_forwarded = []
packets_generated = []

for i in range(NUM_NODES):

    if labels[i] == "normal":
        packets_forwarded.append(packets_sent[i] * np.random.uniform(0.85, 1))
        packets_generated.append(np.random.uniform(5,10))

    elif labels[i] == "selective":
        packets_forwarded.append(packets_sent[i] * np.random.uniform(0.4, 0.7))
        packets_generated.append(np.random.uniform(5,10))

    elif labels[i] == "onoff":
        packets_forwarded.append(packets_sent[i] * np.random.uniform(0.3, 1))
        packets_generated.append(np.random.uniform(5,10))

    elif labels[i] == "blackhole":
        packets_forwarded.append(0)
        packets_generated.append(np.random.uniform(5,10))

    elif labels[i] == "ddos":
        packets_forwarded.append(packets_sent[i] * np.random.uniform(0.8,1))
        packets_generated.append(np.random.uniform(40,60))

packets_forwarded = np.array(packets_forwarded)
packets_generated = np.array(packets_generated)

# ---------------- FEATURE ENGINEERING ----------------
forward_ratio = packets_forwarded / packets_sent
drop_ratio = 1 - forward_ratio
delay = np.random.uniform(0.1, 1.5, NUM_NODES)
energy = np.random.uniform(0.2, 1.0, NUM_NODES)
prev_trust = np.random.uniform(0.3, 0.9, NUM_NODES)

data = pd.DataFrame({
    "forward_ratio": forward_ratio,
    "drop_ratio": drop_ratio,
    "delay": delay,
    "energy": energy,
    "prev_trust": prev_trust,
    "packets_generated": packets_generated,
    "label": labels
})

# ---------------- ML TARGET ----------------
X = data.drop("label", axis=1)
y = (data["label"] != "normal").astype(int)

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ============================================================
# BASELINE MODELS
# ============================================================
baseline_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(5),
    "SVM": SVC(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(150, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "Neural Network": MLPClassifier(max_iter=1000, random_state=42)
}

baseline_results = {}

print("\n========= BASELINE MODEL BENCHMARKING =========\n")

for name, model in baseline_models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    baseline_results[name] = acc
    print(f"{name}: {acc*100:.2f}%")

# ============================================================
# HYPERPARAMETER TUNING FOR TOP 3 MODELS (same accuracy)
# ============================================================

print("\n========= HYPERPARAMETER TUNING =========\n")

# Decision Tree Tuning
print("Tuning Decision Tree...")
dt_params = {
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5, scoring='f1')
dt_grid.fit(X_train, y_train)
dt_tuned = dt_grid.best_estimator_
dt_acc = accuracy_score(y_test, dt_tuned.predict(X_test))
dt_auc = roc_auc_score(y_test, dt_tuned.predict_proba(X_test)[:, 1])
print(f"Decision Tree (Tuned): Accuracy={dt_acc*100:.2f}%, AUC={dt_auc:.4f}")
print(f"  Best params: {dt_grid.best_params_}\n")

# Random Forest Tuning
print("Tuning Random Forest...")
rf_params = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), rf_params, cv=5, scoring='f1', n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf_tuned = rf_grid.best_estimator_
rf_acc = accuracy_score(y_test, rf_tuned.predict(X_test))
rf_auc = roc_auc_score(y_test, rf_tuned.predict_proba(X_test)[:, 1])
print(f"Random Forest (Tuned): Accuracy={rf_acc*100:.2f}%, AUC={rf_auc:.4f}")
print(f"  Best params: {rf_grid.best_params_}\n")

# Gradient Boosting Tuning
print("Tuning Gradient Boosting...")
gb_params = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10]
}
gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_params, cv=5, scoring='f1')
gb_grid.fit(X_train, y_train)
gb_tuned = gb_grid.best_estimator_
gb_acc = accuracy_score(y_test, gb_tuned.predict(X_test))
gb_auc = roc_auc_score(y_test, gb_tuned.predict_proba(X_test)[:, 1])
print(f"Gradient Boosting (Tuned): Accuracy={gb_acc*100:.2f}%, AUC={gb_auc:.4f}")
print(f"  Best params: {gb_grid.best_params_}\n")

# ============================================================
# COMPREHENSIVE MODEL EVALUATION
# ============================================================

print("========= COMPREHENSIVE EVALUATION =========\n")

evaluation_models = {
    "Decision Tree (Baseline)": DecisionTreeClassifier(random_state=42),
    "Decision Tree (Tuned)": dt_tuned,
    "Random Forest (Baseline)": RandomForestClassifier(150, random_state=42),
    "Random Forest (Tuned)": rf_tuned,
    "Gradient Boosting (Baseline)": GradientBoostingClassifier(random_state=42),
    "Gradient Boosting (Tuned)": gb_tuned,
}

eval_results = {}

for name, model in evaluation_models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, 'predict_proba') else 0
    
    eval_results[name] = {
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'auc': auc
    }
    
    print(f"{name}:")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  AUC:       {auc:.4f}\n")

# ============================================================
# SELECT BEST TUNED MODEL
# ============================================================

best_model_name = "Random Forest (Tuned)" if rf_acc > max(dt_acc, gb_acc) else ("Gradient Boosting (Tuned)" if gb_acc >= dt_acc else "Decision Tree (Tuned)")
best_model = evaluation_models[best_model_name]
print(f"\nBEST MODEL SELECTED: {best_model_name}\n")

# ---------------- TRUST HISTORY ----------------
trust_history = []
for t in range(10):
    trust_now = forward_ratio * np.random.uniform(0.8, 1)
    trust_history.append(trust_now)

final_trust = np.mean(trust_history, axis=0)

# ---------------- MALICIOUS DETECTION ----------------
THRESHOLD = 0.5
malicious = np.where(final_trust < THRESHOLD)[0]
normal = np.where(final_trust >= THRESHOLD)[0]

# ---------------- CLUSTER HEAD ----------------
if len(normal) == 0:
    cluster_head = np.argmax(final_trust)
else:
    cluster_head = normal[np.argmax(final_trust[normal])]

print(f"Cluster Head Selected: Node {cluster_head}")

# ================= COMPREHENSIVE VISUALIZATION =================

fig = plt.figure(figsize=(20, 14))

# -------- Row 1: Model Performance Comparison --------

# Fig 1: Baseline Model Accuracy (VERTICAL)
# Fig 1: Baseline Model Accuracy (CLEAN FORMAT)
ax1 = fig.add_subplot(3, 4, 1)

models_list = list(baseline_results.keys())
baseline_accs = list(baseline_results.values())

bars = ax1.bar(models_list, baseline_accs, color='steelblue', width=0.6)

ax1.set_ylabel("Accuracy", fontsize=10)
ax1.set_xlabel("Models", fontsize=10)
ax1.set_title("Fig 1: Baseline Model Accuracy", fontsize=11)

ax1.set_ylim([0, 1])

# Rotate labels properly
ax1.set_xticklabels(models_list, rotation=30, ha='right', fontsize=8)

# Add values on top of bars (important for presentation)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height + 0.01,
             f'{height*100:.1f}%',
             ha='center', va='bottom', fontsize=7)

# Light grid for clarity
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# Fig 2: Tuned Models Performance (already vertical)
ax2 = fig.add_subplot(3, 4, 2)
tuned_models = ['DT', 'RF', 'GB']
tuned_accs = [dt_acc, rf_acc, gb_acc]
colors_tuned = ['#ff7f0e', '#2ca02c', '#d62728']
ax2.bar(tuned_models, tuned_accs, color=colors_tuned)
ax2.set_ylabel("Accuracy")
ax2.set_title("Fig 2: Top 3 Models (Tuned)")
ax2.set_ylim([0, 1])
for i, v in enumerate(tuned_accs):
    ax2.text(i, v + 0.01, f'{v*100:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=8)

# Fig 3: Evaluation Metrics Heatmap (no change)
ax3 = fig.add_subplot(3, 4, 3)
eval_df = pd.DataFrame(eval_results).T
im = ax3.imshow(eval_df.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax3.set_xticks(range(len(eval_df.columns)))
ax3.set_xticklabels(eval_df.columns, rotation=45, ha='right')
ax3.set_yticks(range(len(eval_df.index)))
ax3.set_yticklabels(eval_df.index, fontsize=8)
ax3.set_title("Fig 3: Performance Metrics")
plt.colorbar(im, ax=ax3)

# Fig 4: F1-Score Comparison (VERTICAL)
ax4 = fig.add_subplot(3, 4, 4)
f1_scores = [eval_results[name]['f1'] for name in eval_results.keys()]
ax4.bar(list(eval_results.keys()), f1_scores, color='coral')  # changed barh -> bar
ax4.set_ylabel("F1-Score")
ax4.set_title("Fig 4: F1-Score Across Models")
ax4.set_ylim([0, 1])
ax4.tick_params(axis='x', rotation=45)

# -------- Row 2: Feature Importance Analysis (ALREADY CORRECT) --------

# Fig 5
ax5 = fig.add_subplot(3, 4, 5)
dt_importance = dt_tuned.feature_importances_
ax5.bar(X.columns, dt_importance, color='#ff7f0e')
ax5.set_xlabel("Features")
ax5.set_ylabel("Importance")
ax5.set_title("Fig 5: Decision Tree Feature Importance")
ax5.tick_params(axis='x', rotation=45)

# Fig 6
ax6 = fig.add_subplot(3, 4, 6)
rf_importance = rf_tuned.feature_importances_
ax6.bar(X.columns, rf_importance, color='#2ca02c')
ax6.set_xlabel("Features")
ax6.set_ylabel("Importance")
ax6.set_title("Fig 6: Random Forest Feature Importance")
ax6.tick_params(axis='x', rotation=45)

# Fig 7
ax7 = fig.add_subplot(3, 4, 7)
gb_importance = gb_tuned.feature_importances_
ax7.bar(X.columns, gb_importance, color='#d62728')
ax7.set_xlabel("Features")
ax7.set_ylabel("Importance")
ax7.set_title("Fig 7: Gradient Boosting Feature Importance")
ax7.tick_params(axis='x', rotation=45)

# Fig 8
ax8 = fig.add_subplot(3, 4, 8)
perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
ax8.bar(X.columns, perm_importance.importances_mean, color='purple')
ax8.set_xlabel("Features")
ax8.set_ylabel("Importance")
ax8.set_title(f"Fig 8: Permutation Importance ({best_model_name})")
ax8.tick_params(axis='x', rotation=45)

# -------- Row 3: Confusion & Trust Analysis (no change) --------

# Fig 9
ax9 = fig.add_subplot(3, 4, 9)
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, ax=ax9, cmap='Blues')
ax9.set_title(f"Fig 9: Confusion Matrix ({best_model_name})")

# Fig 10
ax10 = fig.add_subplot(3, 4, 10)
ax10.bar(range(NUM_NODES), final_trust, color='steelblue', alpha=0.7)
ax10.axhline(y=THRESHOLD, linestyle='--', color='red', linewidth=2)
ax10.set_xlabel("Node ID")
ax10.set_ylabel("Trust Value")
ax10.set_title("Fig 10: Trust Distribution")

# Fig 11
ax11 = fig.add_subplot(3, 4, 11)
sc = ax11.scatter(positions[:,0], positions[:,1], c=final_trust, cmap='RdYlGn', s=100, vmin=0, vmax=1)
ax11.scatter(positions[cluster_head, 0], positions[cluster_head, 1], marker='*', s=500, c='gold', edgecolors='black', linewidth=2, label='Cluster Head')
ax11.set_xlabel("X Position")
ax11.set_ylabel("Y Position")
ax11.set_title("Fig 11: Network Deployment")
ax11.legend()
cbar = fig.colorbar(sc, ax=ax11)
cbar.set_label('Trust')

# Fig 12
ax12 = fig.add_subplot(3, 4, 12)
ax12.hist(drop_ratio, bins=20, color='teal', edgecolor='black')
ax12.set_xlabel("Drop Ratio")
ax12.set_ylabel("Frequency")
ax12.set_title("Fig 12: Drop Ratio Distribution")

plt.tight_layout()
plt.savefig("CRSN_All_Results_Optimized.png", dpi=300, bbox_inches='tight')
plt.show()