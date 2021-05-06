# Datasets
[Bank Climate Change](https://www.ran.org/bankingonclimatechange2020/)
* Small dataset that shows Fossil Fuel funding for 36 banks from 2016 - 2019. 

[Sustainable Finance Commitments](https://www.wri.org/finance/banks-sustainable-finance-commitments/)
* Dataset contains detailed overview of bank sustainability commitments, including, but not limited to: commitment amount, whether a timeline is provided and what the timeline is, description of the commitment, etc. 



# Questions/follow up : 
there is a value for applicant sex: "applicant selected both male and female". This value makes up ~0.05% of the dataset. I'm wondering if this was a way that people indicated on the form that they identify as non-binary, or if it was an accidental button click. 35% of adults in the US identify as nonbinary, so certainly the proportions don't line up. Will keep this in the dataset for now, determine how to interpret later. 

Quicken has a large number of nans for the derived sex, ethnicity, and race columns. Will leave this in as a feature to see if not indicating these affects approvals

Removed nearly all co-applicant information. It's well known that adding a co-applicant if you're approval likelihood is low will improve your chances, assuming the co-applicant is qualified. The only information about the co-applicant was protected classes. Will use feature engineering to add "co-applicant minority race, gender, ethnicity" or used the derived version of these from the dataset columns, if it looks like what I need. 

# Missing Values
Because a goal of this project is to streamline data prep-processing, I did a deeper dive into how to handle missing values. According to Yiran Dong and Chao-Ying, a missing value ratio of up to 5% does not affect the bias of the dataset. According to Paul Madley-Dowd, Rachael Hughes, KateTilling, Jon Heron, a missing value ratio of up to 10% is acceptable, so long as the data is missing at random. 
* First, I created a function that would calculate the missing value ratio of all columns and dropped null values that comprised less than 5% of the dataset. 
* Next, I looked at the remaining features with missing value ratios above 5% to determine if values were missing at random. To do this, I created dataframes comprised of the rows containing missing values for each of the features over the acceptable missing value ratio threshold. Then, I compared the summary statistics for each of these dataframes with the summary statistics for the original dataframe. I focused on the mean and the standard deviation. Overall, the missing value subset for census tract, applicant sex, and applicant race-1 showed that the mean and standard deviation were fairly similar across all fields of interest. Conclusion: data appears to be missing at random. 

# Modeling
*Class Imbalance*: To handle class imbalance: 
* Metric chosen: 
    - Training a model on accuracy will simply predict the majority class in an imbalanced dataset. 
    - Training a model using kappa _______
    - ROC-AUC with threshold moving

# Web App Development
### Objectives
* Create an app that predicts the likelihood of approval for ten banks
* Allow users to filter results by environmental stewardship and equitable lending rank, and by loan term conditions

### Lending Institutions
The following ten banks were selected, as they [originated the most loans in 2019](https://www.housingwire.com/articles/here-are-the-top-10-mortgage-lenders-of-2019/), the same year data was pulled: 
* US Bank
* Freedom Mortgage
* Bank of America
* Caliber Home Loans
* loanDepot
* Fairway independent mortgage
* JPMorgan Chase
* Wells Fargo
* United Wholesale Mortgage
* Quicken Loans