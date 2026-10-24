from pathlib import Path
import pandas as pd, joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

ROOT=Path(__file__).parent
df=pd.read_csv(ROOT/"data"/"breast_cancer_dataset.csv")
X=df.drop(columns=["target","target_name"]); y=df["target"].astype(int)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
models={
 "Dummy baseline":DummyClassifier(strategy="most_frequent"),
 "Logistic Regression":Pipeline([("scaler",StandardScaler()),("model",LogisticRegression(max_iter=5000,class_weight="balanced",random_state=42))]),
 "Random Forest":RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced",min_samples_leaf=2,n_jobs=-1)
}
scores={}
for name,model in models.items():
 vals=cross_val_score(model,X_train,y_train,cv=cv,scoring="f1")
 scores[name]=vals.mean()
 print(f"{name}: CV F1={vals.mean():.4f} +/- {vals.std():.4f}")
best_name=max((n for n in scores if n!="Dummy baseline"),key=lambda n:scores[n])
best=models[best_name].fit(X_train,y_train)
pred=best.predict(X_test); prob=best.predict_proba(X_test)[:,1]
print("\nSelected model:",best_name)
print("Accuracy:",round(accuracy_score(y_test,pred),4))
print("Precision (benign):",round(precision_score(y_test,pred,pos_label=1),4))
print("Recall (benign):",round(recall_score(y_test,pred,pos_label=1),4))
print("F1 (benign):",round(f1_score(y_test,pred,pos_label=1),4))
print("ROC-AUC:",round(roc_auc_score(y_test,prob),4))
print(classification_report(y_test,pred,target_names=["malignant","benign"]))
joblib.dump(best,ROOT/"models"/"best_model.joblib")
print("Saved model to models/best_model.joblib")
