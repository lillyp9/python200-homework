import warnings
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(BytesIO(response.content), header=None)
COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]
df.columns = COLUMN_NAMES
print(df['spam_label'].value_counts())
print(df.shape)

print(df.head())

#Notes: Total emails is 4,601
#Ham (not spam): 2,788 (61%)
#Spam 1,813 (39%)

#60/40 split , its a moderately imbalanced 
#Even with the 60/40 split , with the lazy model it would predict 60% accuracy but that number alone could mislead the data. In order to get the right decent accuracy , I would need to use precision , recall, and F1 score.

#key feauture to explore 
features = ['word_freq_free', 'char_freq_!', 'capital_run_length_total']
#create the boxplots
for feat in features:
    plt.figure()
    df.boxplot(column=feat, by='spam_label')
    plt.title(f"{feat} by spam_label")
    plt.suptitle("")
    plt.savefig(f"outputs/{feat}_boxplot.png")
    plt.close()

# The first thing that stood out was that capital_run_length_total had bigger numbers (0 to 16,000) then the other two features (char_freq_!(0 to 30 ), word_freq_free(0 to 20))
#You can tell right away there is a difference between spam and ham. 
# - word_freq_free : ham is close/stay with 0
# - char_freq_! : the spam uses exclamation mark 
# - capital_run_length_total : spam uses CAPITAL LETTERS 

#Theres heavy skew towards zero. Where most feautures are 0 for most emails , only a few non-zero values . 

#Models such as KNN and logistic regression are sensitive to feature scale. 
#Since capital_run_length_total has bigger number which will dominate , if not scaled the other features correctly. Need to do StandardScaler . 

#=========== Task 2 ======
X = df.drop(columns=['spam_label'])
y = df['spam_label']

#Train and test split (80/20) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
 
#Scale features (fit on train not test to avoid leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#===== PCA ====
pca = PCA()
pca.fit(X_train_scaled)
#cumulative variance 
cumulative = np.cumsum(pca.explained_variance_ratio_)
plt.figure()
plt.plot(cumulative)
plt.axhline(0.90, color='red', linestyle='--', label= '90% threshold')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Explained")
plt.title("PCA Variance Explained")
plt.legend()
plt.savefig("outputs/pca_variance.png")
plt.close()

#Find the n the 90%:
n = np.argmax(cumulative >= 0.90) + 1  
print(f"The components for 90% variance: {n}")

#Tranform and slice 
X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca = pca.transform(X_test_scaled)[:, :n]

print(f"X_train_pca shape:{X_train_pca.shape}")
print(f"X_test_pca shape: {X_test_pca.shape}")

# features have to get split first then scale to prevent leakage into the scaler 
#PCA finds the high -variance directions then so any  unscaled feautures with high values would dominate
#fit PCA on train only : same reason as scaler , to prevent leakages 

#=============Task 3 ======

#======= KNN on unscaled data ===
knn_unscaled = KNeighborsClassifier(n_neighbors=5) #trained unscaled data 
knn_unscaled.fit(X_train, y_train)
preds = knn_unscaled.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds)}")
print(classification_report(y_test, preds))

#========= KNN scaled data 
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
preds = knn_scaled.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds)}")
print(classification_report(y_test, preds))

#======= KNn on PCA-reduced data 
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
preds = knn_pca.predict(X_test_pca)
print(f"Accuracy: {accuracy_score(y_test, preds)}")
print(classification_report(y_test, preds))

#Decision Tree 
for depth in [3, 5, 10, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    train_acc = tree.score(X_train, y_train) #train data 
    test_acc = tree.score(X_test, y_test) #test data
    print(f"Depth={depth}: Train={train_acc:.3f}, Test={test_acc:.3f}")

#Decision Tree with max depth (try with 5, try with 10 now)
tree = DecisionTreeClassifier(max_depth=10, random_state=42)
tree.fit(X_train, y_train)
preds = tree.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds)}")
print(classification_report(y_test, preds))
#I noticed that as the depth increase , so does the train and test output. However , train climbes to 100% (1.000), while traning barely get to that. 
#This shows that the tree is memorizing , rather the learning the pattern.
#I picked depth 10 , simply because it has a good balance with the test.
# Random Forest 
rf = RandomForestClassifier(n_estimators=100, random_state = 42)
rf.fit(X_train, y_train)
preds_rf = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds_rf)}")
print(classification_report(y_test, preds_rf))

#=== Logistic Regression on Scaled data 
lr_scaled = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
lr_scaled.fit(X_train_scaled, y_train)
preds = lr_scaled.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, preds)}")
print(classification_report(y_test, preds))


#=== Logistic Regression on PCA 
lr_pca = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
lr_pca.fit(X_train_pca, y_train)
preds = lr_pca.predict(X_test_pca)
print(f"Accuracy: {accuracy_score(y_test, preds)}")
print(classification_report(y_test, preds))
#Out of all models , Random Forest shows 94.5 accuracy.

#Both PCA was worse in both sides. 
#  KNN scaled: 90.7% vs KNN Pca : 45% (huge drop which is a big difference)
# Logistic Regression scaled : 92% , LR PCA 91% (small drop)
#PCA waas suppose to bring more info and help , not a drop of 43% 

# I rather minimize the false postives(legit email marked as spam) to get a better accuracy,
#missing a real email is worse then leting a spam email come through, users can easily discard a spam from their inbox.
#However they wouldnt know if an important email lost in the spam folder.

#print 10 most important feautures for Each 
# Decision Tree 
tree_importances = pd.Series(tree.feature_importances_, index=X.columns)
print(tree_importances.nlargest(10))

#Random Forest
rf_importances = pd.Series(rf.feature_importances_, index=X.columns)
print(rf_importances.nlargest(10))
#Both models  agree with each other mostly , one example char_freq_$ and char_freq_! at the very top , word_freq_remove and word_freq_free. 
#Bar chart for RF 
plt.figure(figsize=(10,6))
rf_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feauture(Random Forest)")
plt.savefig("outputs/feauture_importances.png")
plt.close()

#Confusion matrix 
cm = confusion_matrix(y_test, preds_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Hame', 'Spam'])
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("outputs/best_model_confusion_matrix.png")
plt.close()
#In the confusion matrix, the model makes more false negatives , however its a safe error , even if it lets some spam in, it wont loss the real emails.

#Final Conculsion
#My intuition was correct , where spam emails commonly contain : $$$ for money offer , exclamation mark(!!) , words such as free or remove . All those indicate someone spamming you.

#=====Task 4 
# Cross-Validation for each classifier in Task 5 with cv=5 
#get mean and standard deviation

#KNN unscaled
scores = cross_val_score(KNeighborsClassifier(n_neighbors=5), X_train, y_train, cv=5)
print(f"KNN (Unscaled): mean = {scores.mean():.3f}, std = {scores.std():.3f}")

#KNN scaled 
scores = cross_val_score(KNeighborsClassifier(n_neighbors=5), X_train_scaled, y_train, cv=5)
print(f"KNN (PCA): mean = {scores.mean():.3f}, std={scores.std():.3f}")

#Decision Tree 
scores = cross_val_score(DecisionTreeClassifier(max_depth=10, random_state=42,), X_train, y_train, cv=5)
print(f"Decision Tree : mean={scores.mean():.3f}, std={scores.std():.3f}")
#Random Tree 
scores = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), X_train, y_train, cv=5)
print(f"Random Forest: mean{scores.mean():.3f}, std={scores.std():.3f}")

#Logistic Regression scaled 
scores = cross_val_score(LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'), X_train_scaled, y_train, cv=5)
print(f"Logistic Regression (scaled): mean{scores.mean():.3f}, std={scores.std():.3f}")

#Logistic Regression PCA 
scores = cross_val_score(LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'), X_train_pca, y_train, cv=5)
print(f"Logistic Regression(PCA): mean = {scores.mean():.3f}, std={scores.std():.3f}")

#The most accurate model is Random Forest (0.954) 

#Logistic Regression (PCA) has the lowest std at 0.003 which meaans its barely changed 
#Random Forest std was 0.14 shows stability given that the mean is 0.954 (highes of them all)
#Decision Tree was the least stable 0.019 , shows that a simgle tree is fragile any small cahnges affect them.

#random Forest still remains on top as the highest ,  making it the most accurate.
#Logistic regression comes in second 
#Decision Tree is in the middle 
#while KNn unsclaed is at the bottom 
#For some models the cross validation scores are slightly higher. given that the training fol is cv=5 , compare to 80/20 split.
#CV confirms that Random  Forest is the best model out of all of them.

#=== Task 5 - Pipiline 

#pipeline for Random Forest
rf_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])
rf_pipeline.fit(X_train, y_train)
rf_preds = rf_pipeline.predict(X_test)
print(f"Accuracy: {rf_pipeline.score(X_test, y_test):.3f}")
print(classification_report(y_test, rf_preds))

#pipeline for Logistic regression
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))

])
lr_pipeline.fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_test)
print(f"Accuracy: {lr_pipeline.score(X_test, y_test):.3f}")
print(classification_report(y_test, lr_preds))

#Both are different structure , Random Forest has just the classifier, while Logistic regression has both scaler and classifier. 

#Trees splits on the feature so scaling wont help , Logistic would need scaling , if it doesnt the biggest feauture will take over.

#Pipelines using both preprocessing and model into one object( model), it prevents data leakage simce the  PCA and scaler onlt see the training data.

#Its easy to save and load one object to another.