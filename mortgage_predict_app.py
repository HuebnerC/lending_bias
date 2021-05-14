
from flask import Flask, request
import pickle
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import censusgeocode as cg
import pandas as pd
import numpy as np



# Initialize app
app = Flask(__name__)

# load the pickled models
with open('boa_model.pkl', 'rb') as f:
    boa = pickle.load(f)

with open('wells_model2.pkl', 'rb') as f:
    wells = pickle.load(f)

with open('chase_model.pkl', 'rb') as f:
    chase = pickle.load(f)

with open('USB_model.pkl', 'rb') as f:
    USB = pickle.load(f)

with open('LD_model.pkl', 'rb') as f:
    LD = pickle.load(f)

with open('Fair_model.pkl', 'rb') as f:
    fair = pickle.load(f)

with open('cal_model.pkl', 'rb') as f:
    cal = pickle.load(f)

with open('enc.pkl', 'rb') as f:
    enc = pickle.load(f)

with open('FIPS_dict.pkl', 'rb') as f:
    FIPS_dict = pickle.load(f)


# Home page with form on it to submit new data
@app.route('/')
def get_new_data():
    return '''
        <form action="/predict" method='POST'>
        
          Property address? Please submit in the following format: <br>
          "1600 Pennsylvania Avenue, Washington, D.C. 60039"
        <input type="text" name="address" </input><br><br>

          What is the loan amount?<br>
          <input type="text" name="loan_amount" </input><br><br>
          
          Down payment amount?<br>
          <input type="text" name="down_payment" </input><br><br>

          Length of loan (in years)?<br>
          <input type="text" name = "loan_term" </input><br><br>
          
          What is your pre-tax annual income?<br>
          <input type="text" name = "income" </input><br><br>
          
          What is your total monthly debt? Please include:<br>
          - Monthly rent or house payment<br>
          - Monthly alimony or child support payments<br>
          - Student, auto, and other monthly loan payments<br>
          - Credit card monthly payments (use the minimum payment)<br>
          - Other debts<br>
          <input type="text" name = "monthly_debt" </input><br><br>
          <br><br>
          Select your age: <br>
          <input type="radio" name="age" id=[0] value=[0]> Under 25 </input><br>
          <input type="radio" name="age" id=[1] value=[1]> 25-34 </input><br>
          <input type="radio" name="age" id=[2] value=[2]> 35-44 </input><br>
          <input type="radio" name="age" id=[3] value=[3]> 55-64 </input><br>
          <input type="radio" name="age" id=[4] value=[4]> 65-74 </input><br>
          <input type="radio" name="age" id=[5] value=[5]> Over 74 </input><br>
         <br><br>

          What type of home? <br>
          <input type="radio" name="construction_type" id= [1] value= 1> Site-Built </input><br>
          <input type="radio" name="construction_type" id= [0] value= 0> Manufactored Home </input><br>
          <br><br>
          
          Which best describes your ethnicity?<br>
          <input type="radio" name="ethnicity" id="black" value=[0,1,0]> Hispanic or Latino </input><br>
          <input type="radio" name="ethnicity" id="not_hispan" value=[0,0,0]> Not Hispanic or Latino </input><br>
          <input type="radio" name="ethnicity" id="hispan" value=[0,0,1]> Joint ethnicity </input><br>
          
          Which best describes your race?<br>
          <input type="radio" name="race" id="black" value=[0,0,0,1,0,0,0,0]> Black or African American </input><br>
          <input type="radio" name="race" id="white" value="[0,0,0,0,0,0,0,1]"> White </input><br>
          <input type="radio" name="race" id="native_pacific" value="[0,0,0,0,0,1,0,0]"> Native Hawaiian or Pacific Islander</input><br>
          <input type="radio" name="race" id="two_min_race" value="[0,0,0,0,1,0,0,0]"> Biracial (white) </input><br>
          <input type="radio" name="race" id="two_min_race" value="[1,0,0,0,0,0,0,0]"> Two or more minority races </input><br>
          <input type="radio" name="race" id="asian" value="[0,0,1,0,0,0,0,0]"> Asian </input><br>
          <input type="radio" name="race" id="asian" value="[0,1,0,0,0,0,0,0]"> American Indian or Alaska Native </input><br>
         <br><br>
         
         What is your sex? 
        <input type="radio" name="sex" id="female" value=[0,0,0]> Female </input><br>
        <input type="radio" name="sex" id="joint" value=[1,0,0]> Joint </input><br>
        <input type="radio" name="sex" id="male" value=[0,1,0]> Male </input><br>
        <br><br>
        Type of Loan: 
        <input type="radio" name="loan_type" id="Conventional" value=[1,0,0,0]> Conventional </input><br>
        <input type="radio" name="loan_type" id="FHA" value=[0,1,0,0]> FHA </input><br>
        <input type="radio" name="loan_type" id="VA" value=[0,0,1,0]> VA </input><br>
        <input type="radio" name="loan_type" id="FSA/RHS" value=[0,0,0,1]> FSA/RHS </input><br>

        <br><br><br>
          
          <input type="submit" value="Submit for mortgage approval predictions">
        </form>
        '''

@app.route('/predict', methods = ["GET", "POST"])
def predict():
    # request the text from the form, aggregate and code as needed 
    loan_amount = int(request.form['loan_amount'])
    down_payment = int(request.form['down_payment'])
    loan_to_value_ratio = (loan_amount - down_payment)/loan_amount
    loan_term = (12*(int(request.form['loan_term'])))
    income = int(request.form['income'])
    monthly_debt = int(request.form['monthly_debt'])
    debt_to_income_ratio = (monthly_debt/(income/12))
                                
    # Put DIR in bin
    if debt_to_income_ratio < 20:
        debt_to_income_ratio = [15]
    if debt_to_income_ratio in range(20,30): 
        debt_to_income_ratio  = [25]
    elif debt_to_income_ratio in range(30,36): 
        debt_to_income_ratio = [33]
    elif debt_to_income_ratio in range(50,60):
        debt_to_income_ratio = [55]
    else:
        debt_to_income_ratio = debt_to_income_ratio
        
    applicant_age = int(request.form['age'])

    # Convert address to Census Tract Number, then bin into Census category
    address = request.form['address']
    address = cg.onelineaddress(address, returntype='geographies')
    address = address[0].get('geographies')
    census_tracts = address.get('Census Tracts')[0]
    state = census_tracts.get('STATE')
    county = census_tracts.get('COUNTY')
    tract = census_tracts.get('TRACT')
    tract_category = FIPS_dict.get(str(state+county+tract))
    
    #Convert census_tract to dummy list for model
    census_lst = [0,0,0,0,0,0,0,0,0,0]
    census_lst[tract_category - 1] = 1
    
    conforming_loan_limit = 0
    if loan_amount < 548250: 
        conforming_loan_limit = [1]
    construction_type = int(request.form['construction_type'])
    ethnicity = int(request.form['ethnicity'])
    race = int(request.form['race'])
    sex = int(request.form['sex'])
    loan_type = int(request.form['loan_type'])

    X_user = loan_amount + loan_to_value_ratio + loan_term + income + debt_to_income_ratio + \
                    applicant_age + census_lst + [conforming_loan_limit] + [construction_type] + [ethnicity],[race] + \
                            [sex ]+ \
                        [loan_type] + [construction_type]
      
    return str(X_user)
#   Convert user inputs to be compatible with data preprocessing script
#     if conforming_loan_limit == 'y': 
#         conforming_loan_limit == 1
#     else: 
#         conforming_loan_limit == 0
    
#     if derived_dwelling_category: 
#         derived_dwelling_category == 1 
#     else: 
#         derived_dwelling_category == 0                          
    
                                  
#     if derived_ethnicity: 
#         derived_ethnicity = 'Hispanic or Latino'
#     if derived_ethnicity == 2:
#         derived_ethnicity == 'Not Hispanic or Latino'
#     else: 
#         derived_ethnicity = 'Joint'
                                  
#     derived_race_dict = {1:'American Indian or Alaska Native', 2: 'Asian', 3: 'Black or African American',
#                          4: 'Native Hawaiian or Other Pacific Islander', 5: 'White', 6: '2 or more minority races',
#                          'Joint': 7}
#     derived_race = derived_race_dict.get(derived_race)
    
#     if derived_sex: 
#         derived_sex = 'Male'
#     if derived_sex == 2:
#         derived_sex == 'Female'
#     else: 
#         derived_sex = 'Joint'
                                  
                                  
#     loan_term = 12 * loan_term
    
#     debt_to_income_ratio = monthly_debt/income
                                  
#     # Put DIR in bin
#     if debt_to_income_ratio < 20:
#         debt_to_income_ratio = 15
#     if debt_to_income_ratio in range(20,30): 
#        debt_to_income_ratio  = 25
#     elif debt_to_income_ratio in range(30,36): 
#         debt_to_income_ratio = 33
#     elif debt_to_income_ratio in range(50,60):
#         debt_to_income_ratio = 55
#     else:
#         debt_to_income_ratio = debt_to_income_ratio
    
#     #hardcode age for now, will troubleshoot OrdinalEncoder later
#     applicant_age = np.int64(3.0)
#     X_user_raw = np.array([census_tract, conforming_loan_limit, derived_dwelling_category, derived_ethnicity,
#                       derived_race, derived_sex, loan_type, loan_amount, loan_to_value_ratio, loan_term,
#                       construction_method, income, debt_to_income_ratio, applicant_age])
    
# #     # Create dataframe for user
#     columns = np.array(['census_tract', 'conforming_loan_limit', 'derived_dwelling_category', 
#                     'derived_ethnicity','derived_race', 'derived_sex', 'loan_type', 'loan_amount', 
#                    'loan_to_value_ratio', 'loan_term', 'construction_method','income', 'debt_to_income_ratio', 
#                     'applicant_age'])
#     X_user = pd.DataFrame(X_user_raw.reshape(1,-1), columns= columns)
#     # Not working - Ordinal encoding for age
# #     enc_age_arr = enc.transform(X_user[['applicant_age']].to_numpy())
# #     X_user['applicant_age'] = enc_age_arr
    
    
#     X_user['census_tract'] = X_user['census_tract'].astype(int)
#     X_user['census_tract'] = X_user['census_tract'].astype(str)
#     X_user['census_tract'] = X_user['census_tract'].map(FIPS_dict)
# #     Get dummies not working, hardcoding to check rest of code
# #     categorical_cols = ['census_tract', 'conforming_loan_limit', 'derived_dwelling_category', 'derived_ethnicity', 
# #                         'derived_race', 'derived_sex', 'loan_type', 'construction_method']
                          
# #     X_user = pd.get_dummies(X_user, columns = categorical_cols)
# #     dummies_to_drop = ['derived_sex_Female', 'conforming_loan_limit_NC', 'construction_method_2', 
# #                        'derived_dwelling_category_Single Family (1-4 Units):Manufactured', 
# #                        'derived_ethnicity_Not Hispanic or Latino']
# #     X_user = X_user.drop(dummies_to_drop, axis=1)
    
    
#     sample_user_dummies = np.array([2.55e+05, 1.00e+02, 3.60e+02, 6.20e+01, 4.80e+01, 0.00e+00,
#                                    1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
#                                    0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00, 1.00e+00,
#                                    0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
#                                    0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00, 1.00e+00,
#                                    0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00,
#                                    1.00e+00])
#     # predict on the new data
#     model_lst = [boa, wells, chase, USB, LD, fair, cal]
#     bank_lst = ['Bank of America', 'Wells Fargo', 'JPChase Morgan', 'U.S. Bank', 'Loan Depot', 
#                'Fairway Independent Mortgage', 'Caliber Independent Home Loans']
#     prob_dict = {}
#     for model, bank in zip(model_lst, bank_lst):
#         y_pred = model.predict_proba(sample_user_dummies.reshape(1,-1))[:,1][0]
           
#         prob_dict[bank] = round(y_pred*100, 2)
    
#     return (prob_dict)
    
    
#     # for plotting 
#     X_0 = trainX[trainY == 0] # class 0
#     X_1 = trainX[trainY == 1] # class 1
#     X_2 = trainX[trainY == 2] # class 2
    
#     # color-coding prediction 
#     if Y_pred[0] == 0:
#         cp = 'b'
#     elif Y_pred[0] == 1:
#         cp = 'r'
#     else:
#         cp = 'g'

#     if plt:
#         plt.clf() # clears the figure when browser back arrow used to enter new data

#     plt.scatter(X_0[:, 0], X_0[:, 1], c='b', edgecolors='k', label = 'class 0')
#     plt.scatter(X_1[:, 0], X_1[:, 1], c='r', edgecolors='k', label = 'class 1')
#     plt.scatter(X_2[:, 0], X_2[:, 1], c='g', edgecolors='k', label = 'class 2')
#     plt.scatter(X_n[:, 0], X_n[:, 1], c=cp, edgecolors='k', marker = 'd', \
#         s=100, label = 'prediction')
#     plt.xlabel('Sepal length')
#     plt.ylabel('Sepal width')
#     plt.title('Prediction plotted with training data')
#     plt.legend()
        
#     image = BytesIO()
#     plt.savefig(image)
#     out = image.getvalue(), 200, {'Content-Type': 'image/png'}
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
