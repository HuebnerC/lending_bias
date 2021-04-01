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

<p align="center" >
<img src="images/corr_matrix_2.png" width="400" height="400">
</p>

# Hypothesis Testing 
Prior to exploring the data, I created a few hypotheses and set my alpha value at 0.05. 

### __Hypothesis Test 1: Comparison against national average__

H<sub>0</sub>: Applicants are approved at a rate lower than 92% if their race is not black.

H<sub>a</sub>: Applicants are approved at a lower rate if their race is not white. 

p-value: 0.88

Conclusion: Fail to reject the null hypothesis

### __Hypothesis Test 2: Comparison against dataset average__
H<sub>0</sub>: Wells Fargo applicants are approved at a rate lower than the average if their race is not black.

H<sub>a</sub>: Wells Fargo applicants are approved at a lower rate if their race is not white. 

p-value: 1.0

Conclusion: Fail to reject the null hypothesis


### __Hypothesis Test 3: Using a two-sample test to compare approval rates for white and black applicants:__

H<sub>0</sub>: White applicants are approved for mortgages at a higher rate than black applicants.

H<sub>a</sub>: Black applicants are approved at a lower rate than white applicants.
p-value: 0.06

Conclusion: Fail to reject the null hypothesis
<p align="center" >
<img src="images/sample_mean_dist.png" width="300" height="300">
</p>

# Exploratory Data Analysis
Dataset obtained from [FFEIC](https://ffiec.cfpb.gov/data-publication/documents#modified-lar). 
Interested in 'traditional' home loan approval, and actions related to home ownership
* For this reason, kept the Loan Purpose column

## Observations
* Might need to standardize dataset for geographic region
    * For example, CA makes up 15% of the data, after dropping irrelevant information

* Dataset imbalance: 
    * The dataset contained class imbalance both for the target variable and with the race variables. 




# Building a Predictive Model
### Metric Choice - ROC-AUC
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

While these numbers look promising, I was still concerned about overfitting. I reran the correlation matrix and called the ```feature_importance``` method from the ```RandomForestClassifier```. Using the top features and cross-tabulation, I checked for perfect predictors and then dropped additional variables. 


Other models considered, but took too long to run: 
* KNN
* Gradient Boosted Trees

Linear Regression not considered because of high multicolinearity

## Final Model Metrics
The final model had ~97.5% accuracy. False positives occurred at a rate of 2.3%, false negatives at 0.18%. 

# Evaluating for Bias
In addition to Hypothesis Testing for racial bias in approval rates, I explored denial rates. I focused specifically on Type II errors, which carry the most weight both for the lender and for the applicant. For lenders, a type II error means missed revenue. For applicants, it means missing a huge investment. 

## Process
First, I looked at the rate at which these errors were occuring for white and black applicants in general. False negatives occured at 0.16% for white applicants and 0.40% for black applicants.

Next, I created a copy of the dataset and coded all black applicants as white and all white applicants as black. All other variables were kept the same. Then, I tested the model on the new dummy dataset. 

When white applicants were coded as black, their false denial rates increased by a tenth of a percent. 

When black applicants were coded as white, their false denial rates decreased. Interestingly, the denial rate after the swap is 0.16% - the same rate at which white applicants are incorrectly denied a loan. While there is not evidence of causation, it is interesting that changing a black person’s to white seems to level the playing field,  whereas changing a white person’s race to black does not increase the false denial rate to that of black applicants. This shows that there could still be variables embedded in the dataset that indicate race. 

<p align="center" >
<img src="images/FN_swap_comparison.png">
</p>

# Future Exploration
There is still plenty to be explored on this topic. I would like to: 
 * evaluate other lenders for bias. 
 * re-run the model after dropping all race variables to see how it affects the model accuracy. 
 * try adding a correction factor after the model has run to see if this could be used to offset bias. 
 * use this dataset to predict race instead of loan approval, so I could then look at what the top variables were in determining race and account for those in the loan approval model. 
 * assess bias in other protected classes: ethnicity, sex

 I might also explore the following topics for bias: 
* Pre-approval rates
* Business loan approval
* Reverse mortgage approval