import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



#====== Task 1 : Load and Explore =====
#load the dataset 
data = pd.read_csv('student_performance_math.csv', sep=';')
print(f"Dataset shape: {data.shape}")
print(f"First 5 rows: {data.head(5)}")
print(f"Data types:\n{data.dtypes}")  # prints out the data types of each columns

# histogram
plt.figure(figsize=(10,6))
plt.hist(data['G3'], bins=21, range=(0, 20), edgecolor='black')
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade")
plt.ylabel("Number of Students")
plt.savefig('outputs/g3_distribution.png')
plt.close()

#======= Task 2 : Preprocess the data =======
new_df = data.loc[(data['G3']!=0)].copy()
print(f"Before filtering G3=0: {data.shape}")
print(f"New dataset shape After filtering G3=0: {new_df.shape}")

#keeping these rows with G3=0 will distort the analysis because thye represent the students who failed the course, and including them in the analysis may skew the results and make it harder to identify patterns and trends in the data. By filtering out these rows, we can focus on the students who passed the course and better understand the factors that contribute to their success.

#convert yes/no to columns 1/0 and sex column to 0/1
new_df['schoolsup'] = new_df['schoolsup'].map({'yes': 1, 'no': 0})
new_df['internet'] = new_df['internet'].map({'yes': 1, 'no': 0})
new_df['higher'] = new_df['higher'].map({'yes': 1, 'no': 0})
new_df['activities'] = new_df['activities'].map({'yes': 1, 'no': 0})
#sex solumn
new_df['sex'] = new_df['sex'].map({'M': 1, 'F': 0})
print(new_df[['sex']])
print(new_df[['schoolsup']])
print(new_df[['internet']])
print(new_df[['higher']])
print(new_df[['activities']])

# old dataframe (data) , new dataframe(new_df)
#original correlation between G3 and absences
original_corr = data['G3'].corr(data['absences'])

#new correlations with the new dataframe
new_corr = new_df['G3'].corr(new_df['absences'])
#print results
print(f"Original correlation between G3 and absences: {original_corr}")
print(f"New correlation between G3 and absences: {new_corr}")
#Students with G3=0 is either absence or failed courses, with ut not being remove , the pattern shows having unusual ansence concludes that it doesnt matter. But with the new correlation and filtering , removing G3=0 the pattern immediately shows more absences would result lower grade.

#========== Task 3 : Analyze the data ======
#get the numeric columns  from the new dataframe
numeric_df = new_df.select_dtypes(include='number')

#compute the correlation with G3 for each column
correlations = numeric_df.corr()['G3'].sort_values(ascending=False)
print (f"Correlation of numeric features with G3:\n{correlations}")
#The feautures with the highest postive correlatipn with G3 are G2 and G1 which are the previous grades . The features with the highest negative was schoolsup and failures . The one that did surprise me was the age being in the neagtive feautures.

#Scatter plot 
plt.figure(figsize=(10,6))
plt.scatter(new_df['failures'], new_df['G3'], alpha=0.5)
plt.title("Final Grade (G3) vs Number of Failures")
plt.xlabel("Number of Failures")
plt.ylabel("Final Grade (G3)")
plt.savefig('outputs/g3_vs_failures.png')
plt.close()
#This shows that the number of failures is 0 to 3 , most high grades are at 0 failures , while the students with 3 failures have the lowest grades. This suggests that having more failures is associated with lower final grades, which is consistent with the negative correlation observed between failures and G3.

#Bar chart 
plt.figure(figsize=(10,6))
new_df['schoolsup'].value_counts().plot(kind='bar')
plt.title("Distribution of School Support")
plt.xlabel("School Support (0=No, 1=Yes)")
plt.ylabel("Number of Students")
plt.savefig('outputs/schoolsup_distribution.png')
plt.close()
# This shows that majority of sudents did not reecieve school support , which indictaes the handful that did recieve school support means that have more chalenges faced outside of school , which impacts their academic level and have more absence = lower grades. 

#Baseline Model 
#set up X and Y , use failures alone to print G3
X = new_df[['failures']]
y = new_df['G3']

#split 80/20
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=42)

#Create and fit model
model = LinearRegression()
model.fit(X_train, y_train)

#predit and evaluate the model
y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)
print(f"RMSE: {rmse}")
print(f"Slope: {model.coef_[0]}")
print(f"R2 Score: {r2}")

# The RMSE is 3 which means that the prediction was off by 3 points off which is big significant given the scale is (0-20).
# The R2 is 0.09 which means that faiure alone only explain 9% of the overall grade. Indicating its a weak correlation to use it alone to predict the grade.
#The slope is -1.43 which means for each faiure the grade drops by 1.43 out of the 0-20 sccale.

#======== tASK 5 bUILD fULL mODEL
feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime"]
X = new_df[feature_cols].values
y = new_df["G3"].values

#split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Fit model
model_full = LinearRegression()
model_full.fit(X_train, y_train)

#Evaluate the model
train_r2 = model_full.score(X_train, y_train)
test_r2 = model_full.score(X_test, y_test)
y_pred = model_full.predict(X_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
#print the result 
print(f"Train R2: {train_r2}")
print(f"Test R2: {test_r2}")
print(f"RMSE: {rmse}")
#By adding more features the Train R2 got to 0.175 while the test R2 is 0.154 which increase by 0.06 from the baseline mode R2 Score 0.09
#concludoing it helps but minimal 

#Print each feature with its coefficient 
for name, coef in zip(feature_cols, model_full.coef_):
    print(f"{name:12s}: {coef:+.3f}")

#What surprising is that the coefficent of schoolsup is -2.062 which means that students who get help has lower grades.
#Activities has no effort nor does mother or father education level.
#The Test R2 and Train R2 are close , both are stil considerly low. 
#What should be kept is  schoolsup, faiures , internet , studytime , and sex , this being all those factors that have a significant impact on the grade. The rest of the features should be removed because they have minimal to no impact on the grade such as activities , freetime, medu, traveltime, and fedu.

#==== Scatter Plot of Predicted vs Actual ====
plt.figure(figsize=(10,6))
plt.scatter(y_pred, y_test)

#Diagonal reference line 
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Grade")
plt.ylabel("Actual Grade")
plt.savefig('outputs/predicted_vs_actual_full_model.png')
plt.close()
#The model shows predicted grades are more cluster tightly around 10 -13 , the actaual range from 5 to 18+. The model seems to be playing it safe so even if a student grade is lower or higher , the model would predict the average. 
#Above the diagonal line : model underestimated 
#Below the diagonal line : model overestimated

#====== Final Summary ======
#Filter dataset size 357 rows after reoving G3=0 
#RMSE is 2.86 , which means the model was off by 3 points (round the 2.86)
#R2 is 0.15 which means ony 15% is about the grade , the rest is other factors.
#Largest postive coefficent is internt which means students with access of internet scoore 0.8 points higher.
#Largest negative coefficent is schoolsup (-2.06) , students with support score lower by 2.06 points.

#The surprising result is that parents eduaction both mother and father have no effect on grades. 


#================== Task 7 ===================
#Add G1 to the feature set
feature_cols_with_g1 = feature_cols + ["G1"]

#Set up X and y with G1 included
X = new_df[feature_cols_with_g1].values
y = new_df["G3"].values

#Split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Fit model
model_with_g1 = LinearRegression()
model_with_g1.fit(X_train, y_train)

#print
print(f"Test R2 with G1: {model_with_g1.score(X_test, y_test)}")

#does a high R² here mean G1 is causing G3?
#No the G1 is the first grade ,and  so it doesnt cause the G3 to be higher . 
#   Is this a useful model for identifying students who might struggle?
# This model is useful if we want to predict the final G3 with other factors having the G1 included. Howvere it doesnt help the students who struggle because the G1 is the first grade and if they are struggling in the first grade then they are likely to struggle in the final grade. So it doesnt help us identify students who might struggle before they get their first grade.
#  Early intervention before G1?
# The model from task 5 should be used to identify students who are struggling before they get their G1 grade because it has other factors that can help identify students who might struggle such as school support, absences , studytime , and failures.
