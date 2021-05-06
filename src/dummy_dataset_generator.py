# Evaluating racial bias

def racial_dummyset(processed_df):
    
    # Create array of predicted values on entire test set
    y_pred_base = rf_fitted.predict(X_test)

    # Make df of predictions with predicted and true columns to be used for comparison
    analysis = X_test.copy() 
    analysis['Pred Approval'] = y_pred_base
    analysis['Actual Approval'] = y_test
    
    
    # Create DFs with indices of black applicants and white applicants, respectively. Because I am pulling records
    # by race, I need to keep the True and Predicted values attached to these rows. No longer doing any prediction. 
    black = analysis[analysis['derived_race_Black or African American'] == 1]
    white = analysis[analysis['derived_race_White'] == 1]
    asian = analysis[analysis['derived_race_Asian'] == 1]
    joint_race = analysis[analysis['derived_race_Joint'] == 1]
    race_unavail = analysis[analysis['derived_race_Race Not Available'] == 1]

    # Create confusion matrix for baseline type I, II error rates
    black_conf = confusion_matrix(black[y_test_base], black[y_pred_base])
    white_conf = confusion_matrix(white[y_test_base], white[y_pred_base])
    asian_conf = plus_racial_conf = confusion_matrix(asian[y_test_base], asian[y_pred_base])
    joint_race_conf = confusion_matrix(joint_race[y_test_base], joint_race[y_pred_base])
    race_unavail_conf = confusion_matrix(race_unavail[y_test_base], race_unavail[y_pred_base])

    
    # Creating a copy of test set then switching the race columns for black and white applicants
    X_test_swap = X_test.copy()
    X_test_swap['derived_race_Black or African American'] = X_test_swap['derived_race_Black or African American'] \
                                                                        .map(lambda x: 0 if x ==1 else 1)
    X_test_swap['derived_race_White'] = X_test_swap['derived_race_White'] \
                                                                    .map(lambda x: 0 if x ==1 else 1)
    # Predict on swapped race DF
    swap_pred = rf_fitted.predict(X_test_swap)

    # re-rerun confusion matrix stats
    swap_analysis = X_test_swap.copy()
    swap_analysis['Pred Approval'] = y_pred
    swap_analysis['Actual Approval'] = y_test
    onfusion_matrix(y_true, y_pred)

    # Create dictionary {{black: type_1: num, type_2: num}, 
    #                     {asian: type_1}  }

    race_error_dict = {}
    race_label_lst = ['derived_race_2 or more minority races',
       'derived_race_American Indian or Alaska Native', 'derived_race_Asian',
       'derived_race_Black or African American', 'derived_race_Joint',
       'derived_race_Native Hawaiian or Other Pacific Islander',
       'derived_race_Race Not Available', 'derived_race_White']
    