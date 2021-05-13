import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
# explicitly require this experimental feature

def drop_perfect_predictors(df):
    confirmed_perfect_pred = ['denial_reason-1', 'denial_reason-2', 'denial_reason-3', 'purchaser_type', 'preapproval', 
                       'interest_rate', 'rate_spread', 'hoepa_status', 'total_loan_costs', 'origination_charges', 
                       'discount_points', 'lender_credits', 'initially_payable_to_institution', 'denial_reason-4',
                        'total_points_and_fees']
    df = df.drop(confirmed_perfect_pred, axis=1)

def drop_irrelevant_columns(df): 
    drop_irrelevant = ['state_code', 'county_code', 'open-end_line_of_credit', 'manufactured_home_land_property_interest', 
                  'manufactured_home_secured_property_type', 'co-applicant_credit_score_type', 'applicant_ethnicity-2',
                  'applicant_ethnicity-3', 'applicant_ethnicity-4', 'co-applicant_ethnicity-1', 'co-applicant_ethnicity-2',
                  'co-applicant_ethnicity-3', 'co-applicant_ethnicity-4', 'co-applicant_ethnicity_observed',
                  'applicant_race-2', 'applicant_race-3', 'applicant_race-4', 'applicant_race-5', 
                  'co-applicant_race-1', 'co-applicant_race-2', 'co-applicant_race-3', 'co-applicant_race-4', 
                  'co-applicant_race-5', 'co-applicant_race_observed', 'co-applicant_sex', 'applicant_sex_observed',
                  'co-applicant_sex_observed', 'co-applicant_age', 'applicant_age_above_62', 'co-applicant_age_above_62',
                  'submission_of_application', 'derived_loan_product_type', 'interest_only_payment', 'applicant_ethnicity-1',
                  'applicant_race-1', 'applicant_sex', 'derived_msa-md', 'applicant_ethnicity-5', 
                  'co-applicant_ethnicity-5', 'activity_year', 'lei', 'multifamily_affordable_units', 
                  'aus-2', 'aus-3', 'aus-4', 'aus-5']
    quicken_df = quicken_df.drop(drop_irrelevant, axis=1)