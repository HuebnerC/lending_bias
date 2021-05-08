# Evaluating racial bias

def racial_dummyset(processed_df, fitted_model):
    
    # Create array of predicted values on entire test set
    y_pred_baseline = fitted_model.predict(X_test)

    # Make df of predictions with predicted and true columns to be used for comparison
    analysis = X_test.copy() 
    analysis['Pred Approval'] = y_pred_baseline
    analysis['Actual Approval'] = y_test_baseline
    
    
    # Create DFs with indices of black applicants and white applicants, respectively. Because I am pulling records
    # by race, I need to keep the True and Predicted values attached to these rows. No longer doing any prediction. 
    black = analysis[analysis['derived_race_Black or African American'] == 1]
    white = analysis[analysis['derived_race_White'] == 1]
    asian = analysis[analysis['derived_race_Asian'] == 1]
    joint_race = analysis[analysis['derived_race_Joint'] == 1]
    race_unavail = analysis[analysis['derived_race_Race Not Available'] == 1]

    # Create confusion matrix to determine baseline type I, II error rates
    black_conf = confusion_matrix(black[y_test_baseline], black[y_pred_baseline], normalize = 'all')
    white_conf = confusion_matrix(white[y_test_baseline], white[y_pred_baseline], normalize = 'all')
    asian_conf = plus_racial_conf = confusion_matrix(asian[y_test_baseline], asian[y_pred_baseline], normalize = 'all')
    joint_race_conf = confusion_matrix(joint_race[y_test_baseline], joint_race[y_pred_baseline], normalize = 'all')
    race_unavail_conf = confusion_matrix(race_unavail[y_test_baseline], race_unavail[y_pred_baseline], normalize = 'all')

    
    # Create a copy of processed dataset then switch the race columns for each minority group and white applicants
    # if black == 1, change to 0 and change white to 1
    # if white == 1, change to 0 and change black to 1
    X_test_black = X_test.copy()
    cond = (X_test_black['derived_race_Black or African American'] ==1) | (X_test_black['derived_race_White'] == 1)
    true_race = 'derived_race_Black or African American'
    white = 'derived_race_White'
    X_test_black.loc[cond, [true_race, white]] = X_test_black.loc[cond, [white, true_race]].values
    # Predict on swapped race DF
    pred_swap_black = fitted_model.predict(X_test_black)

    # re-rerun confusion matrix stats
    black_swap_analysis = X_test_black.copy()
    black_swap_analysis['Pred Approval'] = pred_swap_black
    black_swap_analysis['Actual Approval'] = y_test
    black = black_swap_analysis[black_swap_analysis['derived_race_Black or African American'] == 1]
    white = black_swap_analysis[black_swap_analysis['derived_race_White'] == 1]
    black_conf = confusion_matrix(black['Actual Approval'], black['Pred Approval'], normalize= 'all')
    white_conf = confusion_matrix(white['Actual Approval'], white['Pred Approval'], normalize = 'all')
        

    # Create dictionary {{black: type_1: num, type_2: num}, 
    #                     {asian: type_1}  }

    race_error_dict = {}
    race_label_lst = ['derived_race_2 or more minority races',
       'derived_race_American Indian or Alaska Native', 'derived_race_Asian',
       'derived_race_Black or African American', 'derived_race_Joint',
       'derived_race_Native Hawaiian or Other Pacific Islander',
       'derived_race_Race Not Available', 'derived_race_White']
    