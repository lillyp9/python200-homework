import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import requests
import warnings
from io import BytesIO

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay, 
    recall_score,
    f1_score
)
from sklearn.inspection import DecisionBoundaryDisplay
warnings.filterwarnings("ignore", category=RuntimeWarning)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

#============== Preprocessing ======================

#Question 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=42)
print("Train set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])
print("Train set size:", y_train.shape[0])
print("Test set size:", y_test.shape[0])

#Question2 
#fit the scaler on xtrain and use it to transform both xtrain and xtest
scaler = StandardScaler()#Start the scaler 
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(np.mean(X_train_scaled, axis=0)) #should be close to 0
#Scaler was fit on X_train to avoid data leakage. The test set is "new" data
#====== KNN ========
#Question 1
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
preds = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, preds))

#Question 2
scaled = KNeighborsClassifier(n_neighbors=5)
scaled.fit(X_train_scaled, y_train)
scaled_preds = scaled.predict(X_test_scaled)
print("KNN Accuracy with Scaled Data:", accuracy_score(y_test, scaled_preds))
#

#Question 3
knn = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
print(cv_scores)
print(f"Mean: {cv_scores.mean():.3f}")
print(f"Std: {cv_scores.std():.3f}")
#Cross-validation is more reliable then the single split , it evaluates the model on 5 differents panels (train/test splits) reducing the chnaces that one panel/section gives a misleading result.


#Question 4 
k_values = [1,3,5,7,9,11,15]
# The highest mean score is k=5 .. mean=0.975 std = 0.033 , I picked it beacuase there was two 0.975 , but one of them had a higher standard deviation 0.033 , so i slected that option. 

#looping the  k values 
for k in k_values:
    knn =  KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"k={k:2d}: mean={scores.mean():.3f} std{scores.std():.3f}")

#============ Classifier Evaluation =========
#Question 1:
cm = confusion_matrix(y_test, preds) #define the cm 
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(colorbar=False)
plt.title("KNN Confusion Matrix")
plt.savefig("outputs/knn_confusion_matrix.png")
plt.close()
#it looks like versicolor and virginica gets mixed up . 


#================== sklearn API: Decision trees =========
#========Decision Tree 
#Question 1 
tree = DecisionTreeClassifier(max_depth=3, random_state=42)#max-depth=3 to prevent oveffitting, random_state=42 to ensure reproducility
tree.fit(X_train, y_train) #training data 
tree_pred = tree.predict(X_test) #get the predication of x test 
print("Decision Tree Accuracy:", accuracy_score(y_test, tree_pred))
print(classification_report(y_test, tree_pred))
#The Decision Tree Accuracy is 0.96 , and the Knn Accuracy is 1.0, while anaylzing them , they both are close , just 3 points apart.
#Scaling has no effect as the order and value matter more then just whether it was scaled up or down, in past testing , it has little to no difference. 

#=========Logistic Regression =============
#the c values 
C_values = [0.01, 1.0, 100]
#looping the c 
for c in C_values: 
    base_model = LogisticRegression(C=c,max_iter=1000, solver="liblinear")
    model = OneVsRestClassifier(base_model)
    model.fit(X_train_scaled, y_train)
    #get the coefficient 
    coef_sum = sum(np.abs(est.coef_).sum() for est in model.estimators_)
    print(f"C={c}: Total coefficient size = {coef_sum:.3f}")

#=============== PCA =====================
digits = load_digits()
X_digits = digits.data
y_digits = digits.target
images = digits.images

#qUESTION 1 
print("X_digits shape:", X_digits.shape)
print("Images Shape", images.shape)

#1-row subplot 
fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for digit in range(10):
    #find first image 
    idx = np.where(y_digits == digit)[0][0]
    axes[digit].imshow(images[idx], cmap='gray_r')
    axes[digit].set_title(str(digit))
    axes[digit].axis('off')
plt.savefig("outputs/sample_digits.png")
plt.close()


#Question 2 
pca = PCA()
pca.fit(X_digits)
scores = pca.transform(X_digits)

plt.figure()
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)
plt.colorbar(scatter, label='Digit')
plt.title("PCA 2D Projection")
plt.savefig("outputs/pca_2d_projection.png")
plt.close()
#They are all cluster and overlapping one another due to the similarity in digits.

#Question 3 
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Explained")
plt.title("Variance EXplained")
plt.savefig("outputs/pca_variance_explained.png")
plt.close()
# 13 components needed to explain 80% variance 

#Question 4 
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8) 

n_values = [2, 5, 15, 40] #using recontruction through components 
n_digits = 5 #recontrusct

#Get the 5 rows , 5 columns 
fig, axes = plt.subplots(5,5, figsize=(10,10))

#Top rows: Original images
#get the loop
for col in range(n_digits):
    axes[0,col].imshow(images[col], cmap='gray_r')
    axes[0, col].set_title(f"Digit {col}")
    axes[0, col].axis('off')
    axes[0,0].set_ylabel("Original",  fontsize=10)

#Reconstruction for pther rows
for row, n in enumerate(n_values, start=1):
    for col in range(n_digits):
        reconstructed = reconstruct_digit(col, scores, pca , n)
        axes[row, col].imshow(reconstructed, cmap='gray_r')
        axes[row, col].axis('off')
    axes[row, 0].set_ylabel(f"n={n}", fontsize=10)
plt.tight_layout()
plt.savefig("outputs/pca_reconstructions.png")
plt.close()    

#The digitals look clearer around n=15 , row 4 , it matches with the curve around 15-20 components
#13 components was where it cross . 
#n=2 , row 2 was blurry blobs , row 3, n = 5 , started to take shape, but still fuzzy. 