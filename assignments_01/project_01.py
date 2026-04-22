import glob
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from prefect import task, flow, get_run_logger

path = "/home/lilly/python-200/assignments/resources/happiness_project"
output_path = "/home/lilly/python200-homework/assignments_01/outputs"

#@task(retries=3, retry_delay_seconds=2)
#load dataframe 
@task(retries=3, retry_delay_seconds=2)
def load_data():
    #use glob to get a list of all file paths
    file_list = glob.glob(f"{path}/*.csv")
    print(file_list)
    #loop load each file and store in list
    
    all_data =[]
    for file in file_list:
        df = pd.read_csv(file, sep=';', decimal = ',' )
        df['year'] = file.split('_')[-1].split('.')[0]
        all_data.append(df)
        
    #combine all dataframe in list
     
    combine_df = pd.concat(all_data, ignore_index=True)
    combine_df['happiness_score'] = combine_df ['Happiness score'].combine_first(combine_df['Ladder score'])
    print(combine_df['happiness_score'].isna().sum())
   #save the merge data to a csv 
    combine_df.to_csv('/home/lilly/python200-homework/assignments_01/outputs/merged_happiness.csv', index=False)
    return combine_df

#==================================Task 2
@task
def compute_statistics(df):
    logger = get_run_logger()
    logger.info(f"Mean happiness:{df['happiness_score'].mean()}")
    logger.info(f"Median happiness: {df['happiness_score'].median()}")
    logger.info(f"STD happiness: {df['happiness_score'].std()}")

    year_mean = df.groupby('year')['happiness_score'].mean()
    logger.info(f"Year Mean :\n {year_mean}")

    regional_df = df.groupby('Regional indicator')['happiness_score'].mean()
    logger.info(f"Regional:\n{regional_df}")

#=========================Task 3 
@task
def create_visualizations(df):
    logger = get_run_logger()

    plt.hist(df['happiness_score'], bins= 30, color = 'skyblue', edgecolor = 'black')
    plt.title("The happiness Score Overall")
    plt.xlabel('The regionals')
    plt.ylabel('score of happiness')
    plt.savefig(f"{output_path}/happiness_histogram.png")
    plt.close()
    logger.info("Histogram saved successfully")

    #create the boxplot 
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='year', y='happiness_score', data=df)
    plt.title('Happiness Score by Year')
    plt.ylabel('Happiness Score')
    plt.xlabel('Year')
    plt.savefig(f"{output_path}/happiness_by_year.png")
    plt.close()
    logger.info("Boxplot saved successfully")

    #create the scatter plot 
    plt.figure()
    plt.scatter( x =df['GDP per capita'], y = df['happiness_score'])
    plt.title('GDP vs Happiness')
    plt.xlabel('GDP per capita')
    plt.ylabel('Happiness Score')
    plt.savefig(f"{output_path}/gdp_vs_happiness.png")
    plt.close()
    logger.info("Scatter plot saved successfully")

    #create the correlation heatmap
    plt.figure()
    #get the numeric column first 
    numeric_df = df.select_dtypes(include='number')
    sns.heatmap(numeric_df.corr(), annot=True)
    plt.title('Correlation Heatmap')
    plt.savefig(f"{output_path}/correlation_heatmap.png")
    plt.close()
    logger.info("Correlation heatmap saved successfully")

@task
def hypothesis_testing(df):
    logger = get_run_logger()
    
    #testing comparing happiness score from 2019 to 2020 
    group_2019 = df[df['year'] == '2019']['happiness_score']
    group_2020 = df[df['year'] == '2020']['happiness_score']
    
    #run test
    t_stat, p_val = stats.ttest_ind(group_2019, group_2020)

    logger.info(f"T-statistic: {t_stat}")
    logger.info(f"P-value: {p_val}")
    logger.info(f"Mean 2019:{group_2019.mean()}") #get the mean 
    logger.info(f"Mean 2020: {group_2020.mean()}")

    #plain language interpertation of the result at alpha = 0.05
    if p_val < 0.05:
        logger.info("The difference is significant in happiness between 2019 & 2020")
    else:
        logger.info("Theres no significant difference bwteen 2019 & 2020 despite the pandemic") 

    #Second testing 
    regional_1 = df[df['Regional indicator'] == 'South Asia']['happiness_score'] 
    regional_2 = df[df['Regional indicator'] == 'Western Europe']['happiness_score']

    t_stat, p_val2 = stats.ttest_ind(regional_1, regional_2)
    logger.info(f"Regional p-value: {p_val2}")      
   
#Task 5 
@task
def correlation_testing(df):
    logger= get_run_logger()
    #get the numeric colum ns except Happiness_score since it was already done
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'happiness_score']

    #store the results given
    results = []
    for col in numeric_cols:
        #drop Nan values for both columns
        clean = df[['happiness_score', col]].dropna()
        r, p = stats.pearsonr(clean['happiness_score'], clean[col])
        results.append((col, r, p))
        logger.info(f"{col}:  r={r:.3f}, p={p:.4f}") #the r im asking to round 3 decimal places , p rounded to 4 decimal places format string

#bonferroni correction - divide the threshold by number of tests ran
    number_of_tests = len(results)
    adjusted_alpha = 0.05 / number_of_tests
    logger.info(f"\nNumber of tests: {number_of_tests}")
    logger.info(f"Adjusted alpha: {adjusted_alpha:.4f}") #round 4 decimal points 

#which correlations is significant at origina alpha 
    logger.info("\nSignificant at 0.05:")
    for col, r, p in results:
        if p < 0.05:
            logger.info(f"{col}")

#which remian significant after correlation 
    logger.info("\nSignificant after Bonferroni")
    for col , r, p in results:
        if p < adjusted_alpha:
            logger.info(f"{col}")
            
#======= Task 6
@task
def summary_report(df):
    logger = get_run_logger()
    total_countries = df['Country'].nunique()
    total_year = df['year'].nunique()
    logger.info(f"Total countries: {total_countries}")
    logger.info(f"Total years: {total_year}")

    #top and bottom 3 regions by mean happiness score 
    regional_means = df.groupby('Regional indicator')['happiness_score'].mean()
    logger.info(f"\nTop 3 regions:\n{regional_means.nlargest(3)}")
    logger.info(f"\nBottom 3 regions: \n{regional_means.nsmallest(3)}")

    #the result of pre/post-2020 t-test in plain language
    logger.info("\n2019 vs 2020: No significant difference despite the pandemic")

    #strongest correlation 
    logger.info("Strongest correlation: GDP per capita")
        
    
@flow
def happiness_pipeline():
    df = load_data()
    compute_statistics(df)
    create_visualizations(df)
    hypothesis_testing(df)
    correlation_testing(df)
    summary_report(df)


if __name__ == "__main__":
    happiness_pipeline()
    