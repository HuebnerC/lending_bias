# Overview
In the U.S., systemic racism persists and acts as a barrier to protected classes. One of the most impactful ways to accumulate generational wealth is through home ownership. It’s widely known that computer models help humans make decisions - whether it’s what movie to watch, who to follow on social media, or who to approve for a home loan. In many cases, what businesses seek to do is use the hard, cold data to remove subjectivity. This aim is often well-intentioned: make more equitable decisions. However, computer models are only as good as the people creating them. Unintentional bias can creep in. Banking institutions specifically, and recently, have paid large settlements after they were found to have made biased lending decisions. 

In this project, I will attempt to create a predictive home loan approval model that accounts for racial bias, if present. 

# Datasets used
Three datasets were used in this project. 
The primary dataset was obtained from [FFEIC](https://ffiec.cfpb.gov/data-publication/documents#modified-lar). There were over 1 million rows and 58 columns. Data pulled was from Wells Fargo, but FFEIC offers the ability to pull data from other lending institutions as well. 

The second dataset was obtained from [USDA: Economic Research Service](https://www.ers.usda.gov/data-products/rural-urban-commuting-area-codes.aspx) and was used to map Census Tract IDs in the primary dataset, of which there were ~65,000 unique values, to 10 primary codes indicating whether the location was rural, urban, etc. Definitions of these location codes are found [here](https://depts.washington.edu/uwruca/ruca-codes.php).

The third dataset was obtained from [US Census](https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.2018.html), and was used to map latitude and longitude to census tract IDS. 

The data came with a few challenges. First, Wells Fargo approved loans more than the national average. The data is self-reported by the bank - some things could have been omitted for privacy purposes or other reasons. Or maybe this bank just markets to higher-qualified clientele. Why this discrepancy exists isn’t readily answered, so it’s worth noting. 

Credit scores, a big factor in approval decisions, had been scrubbed by the bank before I downloaded the data. 


# Data Preprocessing
The data contained categorical, ordinal, and continuous variables. Prior to exploring the data, I performed some general dataset cleanup: 

__*Null Values*__

Many null values were dropped. For others, I chose to impute the median of the column, as the average was skewed by outliers in some cases. In other cases, I elected to use the mode. These choices are documented in the Mortgage Lending Jupyter Notebook. 

__*Categorical Variables*__

I used ordinal encoding for the age column because because there is a ranked order between age groups.  

Debt-to-Income Ratio column: there were both integers and ranges. It seemed odd to have the ratios broken out this way, so I'm assuming these values are important for mortgage, point for point, above a certain threshold. For this reason, it seemed important to keep the individual values between 36-50%. For the interval values, I encoded these to be the median of the interval. 

*Multicolinearity*

After initial preprocessing, I ran a few baseline machine learning models. The recall score was 100% which indicated there could be data leakage and/or overfitting. I dropped several additional features after confirming that they were associated with the target. For example, 'Reason for Denial' was only nan for loans that were approved, whereas 'Origination Charges' were only nan for loans that were denied. 

I generated a correlation matrix with the remaining features and created a dictionary that listed the colinear features with a value greater than 0.75. I then compared these pairs against the correlation with the target using crosstabulation and dropped the feature that was less correlated. 

# Hypothesis Testing 
Prior to exploring the data, I created a few hypotheses and set my alpha value at 0.05. 

__*Hypothesis Test 1: Comparison against national average*__

H<sub>0</sub>: Applicants are approved at a rate lower than 92% if their race is not black.

H<sub>a</sub>: Applicants are approved at a lower rate if their race is not white. 

p-value: 0.88

Conclusion: Fail to reject the null hypothesis

__*Hypothesis Test 1: Comparison against dataset average*__
H<sub>0</sub>: Wells Fargo applicants are approved at a rate lower than the average if their race is not black.

H<sub>a</sub>: Wells Fargo applicants are approved at a lower rate if their race is not white. 

p-value: 1.0

Conclusion: Fail to reject the null hypothesis


__*Hypothesis Test 3: Using a two-sample test to compare approval rates for white and black applicants:*__

H<sub>0</sub>: White applicants are approved for mortgages at a higher rate than black applicants.

H<sub>a</sub>: Black applicants are approved at a lower rate than white applicants.
p-value: 0.06

Conclusion: Fail to reject the null hypothesis

<p align="center">
  <img src="https://github.com/HuebnerC/lending_bias/blob/master/corr_matrix_2.png" alt="Size Limit CLI" width="738">
</p>
![Test](images/sample_mean_dist.png)

# Exploratory Data Analysis
Dataset obtained from [FFEIC](https://ffiec.cfpb.gov/data-publication/documents#modified-lar). 
Interested in 'traditional' home loan approval, and actions related to home ownership
* For this reason, kept the Loan Purpose column

## Observations
* Might need to standardize dataset for geographic region
    * For example, CA makes up 15% of the data, after dropping irrelevant information

* Dataset imbalance


# Future Exploration
* Pre-approval rates
* Business loan approval
* Reverse Mortgage approval

# Building a Predictive Model
### Metric Choice - Recall
In choosing which metric to optimize, I considered what it might cost lenders to miss out on a qualified applicant and assigned a "risk" value to lending to an unqualified applicant. 
* When interpreting model, select or create a profit measure 
    * Average ROI for lenders - what would missing out on qualified applicants cost? How much missed revenue?
        * Assume an interest of 3.5%, 30 year fixed mortgage
    * What is the cost of lending to an unqualified applicant? 
        * Average foreclosure rate
        * Average percent loss if goes to foreclosure

According to the [FDIC](https://www.fdic.gov/about/comein/files/foreclosure_statistics.pdf), the average lender loses approximately $50,000 per foreclosure, and approximately 1 in 200 homes will go to foreclosure. While this is an oversimplification, for the purpose of this study, I'll assume that a loan granted to an unqualified applicant (false positive or type II error) will have a potential cost of $50,000. 

In 2018, the [median home price](https://www.housingwire.com/articles/average-imb-made-over-5500-in-profit-per-loan-in-q3/) in the US was approximately $255,000. Assuming a 30 year mortgage, interest rate of 3.5%, and [average down payment of 6%](https://www.rocketmortgage.com/learn/what-is-the-average-down-payment-on-a-house#:~:text=The%20average%20down%20payment%20in,loan%20or%20a%20VA%20loan.), using an online calculator, the total interest paid over the life of a loan is $147,789.64. For simplicity, I'll assume lenders make an average profit of approximately $145,000 per loan. 


 True Class
| Positive      | Negative|
| ----------- | ----------- |
| TP: predicted approved actually approved + $145K | FP: predicted approved actually denied -$50K|
| TN: predicted denied actually denied -$0  | FN: app predicted denied was actually approved -$145K|

Considering the lost revenue opportunity is greater than the cost risk associated with approving an unqualified applicant, I chose to optimize recall, which is used to minimize false negatives. While I was approaching this question from a purely financial lense, minimizing recall might also help negate bias against unprotected classes of people. 

## Subsetting the dataset
After data preprocessing, there were approximately 700,000 entries. After attempting to run a few baseline models overnight, I found this to be too computationally expensive and decided to select a subset of the data by taking every 200th entry, resulting in 1,994 entries. I verified that the class distribution was approximately the same as with the original dataset. The training subset yields ~ 148 entries per feature, and the testing subset ~ 44 entries per feature.

## Baseline models produced the following recall scores:
|Model Used |Subset scores| Full test set scores|
| --------------| --------------| ----------------------------|
| SGD Classifier  | 94.28%     |94.24%
| Random Forest    |99.65%     | 99.74%
| Logistic Regression  |98.33%  | 97.91%

While these numbers look promising, I was still concerned about overfitting. 


Other models considered, but took too long to run: 
* KNN
* Gradient Boosted Trees

Linear Regression not considered because of high multicolinearity

# To-do
* Calculate profit/loss for different scenarios, keep it simple with a bar chart
    * excluding protected class features altogether
    * Adding a correction after the fact, using protected class features as a flag
    * Run the model again for a different lending institution
    * Create a heatmap of approvals using census tract