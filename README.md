# Overview
In the U.S., systemic racism persists and acts as a barrier to protected classes. One of the most impactful ways to accumulate generational wealth is through home ownership. In this project, I will attempt to create a predictive home loan approval model that accounts for racial bias, if present. 

# Hypothesis Testing 

H_0: Applicants are approved for mortgages at the same rate regardless of race.

H_a: Applicants are approved at a lower rate if their race is not white. 

# Exploratory Data Analysis
Dataset obtained from FFEIC. 
Interested in 'traditional' home loan approval, and actions related to home ownership
* For this reason, kept the Loan Purpose column

## Observations
* Might need to standardize dataset for geographic region
    * For example, CA makes up 15% of the data, after dropping irrelevant information

# Future Exploration
* Pre-approval rates
* Business loan approval
* Reverse Mortgage approval

# Building a Predictive Model
### Metric Choice
* When interpreting model, select or create a profit measure 
    * Average ROI for lenders - what would missing out on qualified applicants cost? How much missed revenue?
        * Assume an interest of 3.5%, 30 year fixed mortgage
    * What is the cost of lending to an unqualified applicant? 
        * Average foreclosure rate
        * Average percent loss if goes to foreclosure