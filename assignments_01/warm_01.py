import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prefect import task, flow 
from scipy import stats
from scipy.stats import pearsonr

#====== Panda Q1 
data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

print(df.head(3))
print(df.shape)
print(f"Name Rows: {len(df['name'])}")
print(f"Grade Rows: {len(df['grade'])}")
print(f"City Rows: {len(df['city'])}")
print(f"Passed Rows: {len(df['passed'])}")

#====== Panda Q2
#student that passed AND have grade above 80
passed_student = df[(df['grade'] > 80 ) & (df['passed'])]
print(passed_student)

#====== Panda Q3
df["grade_curved"] = df["grade"] + 5
print(df)

#===== Panda Q4
df['name_upper'] = df['name'].str.upper()
print(df [['name', 'name_upper']])

#===== Panda Q5

result = df.groupby('city')['grade'].mean()
print(result)

#=== Panda Q6
df['city'] = df['city'].replace({'Austin':'Houston'})
print(df[['name', 'city']])

#===== Panda Q7
df.sort_values(by='grade', ascending=False)
print(df.head(3))

#=================== NumPy Review ===================
#===== NumPy Q1
arr = np.array([10, 20, 30, 40, 50])
print(arr.shape)
print(arr.dtype)
print(arr.ndim)

#===== NumPy Q2
arr2 = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print(arr2.shape)
print(arr2.size)

#===== NUmpy Q3
new_block = arr2[:2, :2]
print(new_block)

#=====Numpy Q4
zeros = np.zeros((3, 4))
ones = np.ones((2, 5))
print(zeros)
print(ones)

#===== Numpy Q5
array = np.arange(0, 50, 5)
print(array.shape)
print(array.mean())
print(array.sum())
print(array.std())

#===== Numpy Q6
array = np.array(np.random.normal(200))
print(array.mean())
print(array.std())

#=====================Mataplotlib Review======================
#===== Matplotlib Q1
x=[1, 2, 3, 4, 5],
y=[1, 4, 9, 16, 25],
plt.plot(x, y, color='blue', marker='o', linestyle='--', label='Squares')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Squares')
plt.savefig('squares.png')
plt.close()

#===== Matplotlib Q2
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]

plt.bar(subjects, scores, color=['red', 'green', 'blue', 'orange'])
plt.xlabel('Subjects')
plt.ylabel('Scores')
plt.title('Subject Scores')
plt.savefig('subject_scores.png')
plt.close()

#===== Matplotlib Q3
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.scatter(x1, y1, color='blue', label='Dataset 1')
plt.scatter(x2, y2, color='red', label='Dataset 2')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot')
plt.savefig('scatter_plot.png')
plt.close()

#===== Matplotlib Q4
fig , (Q1, Q2)  = plt.subplots(1, 2, figsize=(10, 5))
Q1.plot(x, y, color='blue', marker='o', linestyle='--')
Q1.set_title('Line Plot')
Q2.bar(subjects, scores, color=['red', 'green', 'blue', 'orange'])
Q2.set_title('Bar Chart')
plt.tight_layout()
plt.savefig('subplots.png')
plt.close()


#===== Descriptive Q1
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
print(f"Mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Variance: {np.var(data)}")
print(f"Standard Deviation: {np.std(data)}")

#------ Descriptive Q2
value = np.random.normal(65, 10, 500)
print(f"Mean: {np.mean(value)}")
print(f"Standard Deviation: {np.std(value)}")

plt.hist(value, bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.savefig('histogram.png')
plt.close()

#===== Descriptive Q3
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.boxplot([group_a, group_b], labels=['Group A', 'Group B'])
plt.title('Score Comparison')
plt.ylabel('Score')
plt.savefig('boxplot.png')
plt.close()

#===== Descriptive Q4
normal_data = np.random.normal(50, 5, 200)
#mean is 50 , standard deviation is 5 
skewed_data = np.random.exponential(10, 200)
data_plot = [normal_data, skewed_data]

plt.boxplot(data_plot, labels=['Normal', 'Exponential'])
plt.title('Distribution Comparison')
plt.ylabel('Value')
plt.savefig('boxplot2.png')
plt.close()
#which distrubtion is more skewed ?
#exponential distribution is more skewed than normal distribution because it has a longer tail on the right side, while the normal distribution is symmetric around its mean. 

#which descriptive statistics would provide more appropriate measure of central tendency for each distrubution ?
#For normal distribution using the mean is more appropriate since its symmetric and isnt affected by outliers.
#while exponential distribution  is skewed  so using the median is more appropriate, given that the skew pulls the mean  towards the tail. 

#===== Descriptive Q5
data1 = [10, 12, 12, 16, 18]

data2 = [10, 12, 12, 16, 150]

print(f"Data1 Mean: {np.mean(data1)}")
print(f"Data1 Median: {np.median(data1)}")
print(f"Data1 Mode: {pd.Series(data1).mode()[0]}")

print(f"Data2 Mean: {np.mean(data2)}")
print(f"Data2 Median: {np.median(data2)}")
print(f"Data2 Mode: {pd.Series(data2).mode()[0]}")

#data 2 has an outlier (150) compare to data 1 that has an outlier(18) , clearly shows that data2 has a bigger outlier changing the mean significantly while the median and mode remain unaffected.

#=======================Hypothesis Testing 

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_statistic, p_value = stats.ttest_ind(group_a, group_b)
print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")

#============ Hypothesis Testing Q2
if p_value < 0.05:
    print("Reject the null hypothesis: There is a significant difference between the two groups.")
else:    
    print("Fail to reject the null hypothesis: There is no significant difference between the two groups.")


#====== Hypothesis Testing Q3
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

t_statistic, p_value = stats.ttest_rel(before, after)
print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")

#===== Hypothesis Testing Q4
scores = [72, 68, 75, 70, 69, 74, 71, 73]
t_statistic, p_value = stats.ttest_1samp(scores, 70)
print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")

#===== Hypothesis Testing Q5
group_a = [72, 68, 75, 70, 69,73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_statistic, p_value = stats.ttest_ind(group_a, group_b, alternative='less')
print(f"P-value: {p_value}")

#========= Hypothesis Testing Q6
print("The The p-value went up by 6 points , started off as 1.54 , change to 7.73 after changing the alternative hypothesis to 'less' , this is because the 'less' alternative hypothesis tests if the mean of group_a is less than the mean of group_b. Indicating that the rejected null hypoothesis is for the difference , in compare to the aternative hypothesis ")

#======================= Correlation Review =======================

#===== Correlation Q1
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

corr_matrix = np.corrcoef(x, y)
print(corr_matrix[0, 1])

print("I would think the correlation be 0.5 , since its inbetween 0 and 1, but given its 0.9 that means there a strong relationship between (x, y)")

#====== Correlation 02 =======
x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

r, p = pearsonr(x,y)

print(f"Correlation Coefficent:{r}")
print(f"P-value: {p}")


#======== Correlation Q3
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
corr_matrix = df.corr()

print("Correlation Matrix:")
print('corr_matrix')


#========= Correlation Q4 
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

plt.scatter(x,y,color='red')
plt.title("Negative Correlation")
plt.xlabel("Lower numbers")
plt.ylabel("higher number")
plt.savefig('scatter_plot.png')



#======= Correlation Q5
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation People Map")
plt.savefig('correlation.map.png')
plt.close()

#=============== Pipline Question

def create_series(arr):
    return pd.Series(arr, name="values")

def clean_data(series):
    cleaned_series = series.dropna()
    return cleaned_series

def summarize_data(series):
    summary = {
        "mean" : series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }
    return summary 

def data_pipeline(arr):
    series = create_series(arr)
    clean = clean_data(series)
    summary = summarize_data(clean)
    return summary
#Main Data to Execute 
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

result = data_pipeline(arr)

#print results 
for key , values in result.items():
    print(f"{key} :{values}")

