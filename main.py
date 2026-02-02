import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("loan_approval_dataset.csv",index_col=0)
df.columns = df.columns.str.strip()
df['loan_status'] = df['loan_status'].str.strip()
#print(df.info()

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,f1_score,classification_report,confusion_matrix
X=df.drop(columns=['loan_status'],axis=1)
y = df['loan_status'].map({'Rejected': 0,'Approved': 1})
catcols=X.select_dtypes(include='object').columns
numcols=X.select_dtypes(include='int64').columns
preprocessing=ColumnTransformer(
    transformers=[
        ('cat',OneHotEncoder(),catcols)
    ],
)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
treemodel=Pipeline(steps=[
    ('preprocessor',preprocessing),
    ('classifier',DecisionTreeClassifier(splitter='best',criterion='entropy',class_weight='balanced',max_depth=8,random_state=42))
])
treemodel.fit(X_train,y_train)
y_pred=treemodel.predict(X_test)
y_proba=treemodel.predict_proba(X_test)[:,1]
print(f"Accuracy : {accuracy_score(y_test,y_pred)}")
print(f"f1 score : {f1_score(y_test,y_pred)}")
print(f"roc_auc_score : {roc_auc_score(y_test,y_proba)}")
print(f"classification report : \n{classification_report(y_test,y_pred)}")
print(f"confusion matrix : \n{confusion_matrix(y_test,y_pred)}")


from sklearn.tree import plot_tree
tree = treemodel.named_steps['classifier']
ohe = treemodel.named_steps['preprocessor'].named_transformers_['cat']
encoded_cat_features = ohe.get_feature_names_out(catcols)
feature_names = list(encoded_cat_features) + list(numcols)
plt.figure()
plot_tree(
    tree,
    feature_names=feature_names,
    class_names=['Rejected', 'Approved'],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.show()
fpr,tpr,thresholds=roc_curve(y_test,y_pred)
plt.figure()
plt.plot(fpr,tpr,label='area = %0.2f'%roc_auc_score(y_test,y_pred))
plt.plot([0,1],[0,1],'r--')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC CURVE')
plt.show()