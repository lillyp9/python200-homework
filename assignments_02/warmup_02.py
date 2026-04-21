import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model = LinearRegression()
model.fit(years, salary)
#predicts salary for someone with certain years of experience
years_to_predict = np.array([4,8]).reshape(-1,1)
salaries_predict = model.predict(years_to_predict)

print(f"Slope: {model.coef_[0]}")
print(f"Intercept:{model.intercept_}")
print(f"Prediction for 4 years: {salaries_predict[0]}")
print(f"Prediction for 8 years: {salaries_predict[1]}")

#question2 
x = np.array([10, 20, 30, 40, 50])

#convert x to 2D
x_shaped = x.reshape(-1,1)
print(x_shaped)

#Sckit-learn needs x to be 2D beacuase first 2D treats the info like a table that even if there is one catergory , it will still know where to look for the data. X is needed to make sure the data is structure  likea  table, spreadsheet, so evvery piece of information has a specific spot.

#Question 3 

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_clusters)
predict_clusters = kmeans.predict(X_clusters)
print(f"Cluster Centers:\n{kmeans.cluster_centers_}")
print(f"Points fall in clusters:\n{np.bincount(predict_clusters)}")

#scatter plot of the clusters
plt.scatter(X_clusters[:,0],X_clusters[:,1], c=predict_clusters, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.savefig('outputs/cluster_plot.png')
plt.close()

#========== Linear Regression ==========
np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)
#Question 1 
plt.scatter(age, cost, c=smoker,cmap="coolwarm")
plt.xlabel("Age")
plt.ylabel("Cost")
plt.title('Medical Cost vs Age')
plt.savefig('outputs/cost_vs_age.png')
plt.close()
# The scatter plot shows two groups , smokers being the red have a higher cost thean non smokers which is blue . Its suggesting that being a smoker is a strong predictor of medical cost.

#Question 2 

#reshape age to a 2D arraybefore using x
x = age.reshape(-1,1)
#split data, train & test sets using age and 80/20 split 
x_train, x_test , y_train, y_test = train_test_split(x, cost, test_size=0.2, random_state=42)

print(f"X_train shape: {x_train.shape}")
print(f"X_test shape: {x_test.shape}")
print(f"Y_train shape: {y_train.shape}")
print(f"Y_test shape: {y_test.shape}")

#===Question 3 ===
model= LinearRegression()
model.fit(x_train, y_train)
#print the slope and intercept 
print(f"Slope:{model.coef_[0]}")
print(f"Intercept:{model.intercept_}")
#predict the test set 
x_predict = model.predict(x_test)
y_predict = model.predict(x_test)
print(f"Predicted costs for test set: {x_predict}")
print(f"Actual costs for test set: {y_predict}")
print(f"RmSE: {np.sqrt(np.mean((y_predict - y_test) ** 2))}")
print(f"R2 on test score: {model.score(x_test, y_test)}")
#The slope is 196.57 every extra year of age the person medical cost goes up by $196.57


#=== Question 4 ==
X_full = np.column_stack([age, smoker])
#Split data 
x_train, x_test, y_train, y_test = train_test_split(X_full, cost, test_size=0.2, random_state=42)
#Create and fit model
model_full = LinearRegression()
model_full.fit(x_train, y_train)

#score on test set 
r_squared = model_full.score(x_test, y_test)
print(f"Test R: {r_squared}")

#print  coefficents 
print(f"Age coefficient: " , model_full.coef_[0])
print(f"Smoker coefficient: " , model_full.coef_[1])
#What this mean is that every year the medical cost increases by $205 regardless whether the patient is a smoker or nonsmoker. However being a smoker the cost is $14,538 higher than a non-smoker.

#== Question 5 ==
plt.scatter(y_predict , y_test, color='teal')

#min and max to define the line and diagonal line 
min_val = min(y_test.min(), y_predict.min())
max_val = max(y_test.max(), y_predict.max())

#now plot the line
plt.plot([min_val, max_val], [min_val, max_val], 'r--')

plt.title("Predicted vs Actual")
plt.xlabel("Predicted Costs")
plt.ylabel("Actual Costs")
plt.savefig('outputs/predicted_vs_actual.png')
plt.close()

# the diagonal line is what the prediction of the cost is , above the diagonal is the rise of cost indicating that the model underestimated the true cost . the  below the diagonal line shows that the model overestimated the cost. the diagonal is the right prediction

