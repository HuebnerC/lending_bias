
from flask import Flask, request
import pickle
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import censusgeocode as cg



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


# Home page with form on it to submit new data
@app.route('/')
def get_new_data():
    return '''
        <form action="/predict" method='POST'>
          Property Address in this format: '1600 Pennsylvania Avenue, Washington, DC'<br>
          <input type="text" name="address"> 
          <br><br>
          Is the loan amount greater than $548,250? (y or n) <br>
          <input type="text" name="conforming_loan_limit"> 
          <br><br>
          1 - Manufactured home or 2 - site-built?:<br>
          <input type="text" name="derived_dwelling_category"> 
          <br><br>
          1 - Manufactured home or 2 - site-built?:<br>
          <input type="text" name="construction_method"> 
          <br><br>
          Applicant ethnicity: 1- Hispanic Hispanic or Latino 2- Not Hispanic or Latino 3- Joint<br>
          <input type="text" name="derived_ethnicity"> 
          <br><br>
          Applicant race: 1- American Indian or Alaska Native 2- Asian 3- Black or African American
          4- Native Hawaiian or Other Pacific Islander 5- White 6- 2 or more minority races 7- Joint<br>
          <input type="text" name="derived_race"> 
          <br><br>
          Applicant sex: 1- Male 2- Female 3- Joint<br>
          <input type="text" name="derived_sex"> 
          <br><br>
          Loan type: 1 - Conventional 2- FHA 3- VA 4- USDA<br>
          <input type="text" name="loan_type"> 
          <br><br>
          Loan amount: <br>
          <input type="text" name="loan_amount"> 
          <br><br>
          Down payment amount:<br> 
          <input type="text" name="down_payment"> 
          <br><br>
          Loan Term: Enter number of years amount:<br> 
          <input type="text" name="loan_term"> 
          <br><br>
          Monthly Pre-tax Income:<br>
          <input type="text" name="income">
          <br><br>
          Monthly debt - Include: Monthly rent or house payment, Monthly alimony or child support payments, 
          Student, auto, and other monthly loan payments, 
          Credit card monthly payments (use the minimum payment), Other debts<br>
          <input type="text" name="monthly_debt"
          <br><br>
          Applicant age:<br>
          <input type="text" name="applicant_age">
          <br><br>
          <br>
          <input type="submit" value="Submit for mortgage approval predictions">
        </form>
        '''

@app.route('/predict', methods = ["GET", "POST"])
def predict():
    # request the text from the form 
    address = str(request.form['address'])
    conforming_loan_limit = str(request.form['conforming_loan_limit'])
    derived_dwelling_category = float(request.form['derived_dwelling_category'])
    derived_ethnicity = float(request.form['derived_ethnicity'])
    derived_race = float(request.form['derived_race'])
    derived_sex = float(request.form['derived_sex'])
    loan_type = float(request.form['loan_type'])
    loan_amount = float(request.form['loan_amount'])
    down_payment = float(request.form['down_payment'])
    loan_term = float(request.form['loan_term'])
    construction_method = float(request.form['construction_method'])
    income = float(request.form['income'])
    monthly_debt = float(request.form['monthly_debt'])
    applicant_age = float(request.form['applicant_age'])
    
#   Convert address to Census Tract Number
    address = cg.onelineaddress(str(address), returntype='geographies')
    address = address[1].get('geographies')
    census_tracts = address.get('Census Tracts')[0]
    state = census_tracts.get('STATE')
    county = census_tracts.get('COUNTY')
    tract = census_tracts.get('TRACT')
    census_tract = int(state+county+tract)
    
#   Convert user inputs to be compatible with data preprocessing script
    if conforming_loan_limit.isin(['y', 'yes', 'Yes', 'Y']): 
        conforming_loan_limit == 'C'
    else: 
        conforming_loan_limit == 'U'
    
    if derived_dwelling_category: 
        derived_dwelling_category == 'Single Family (1-4 Units):Manufactured' 
    else: 
        derived_dwelling_category == 'Single Family (1-4 Units):Site-Built'                           
    
                                  
    if derived_ethnicity: 
        derived_ethnicity = 'Hispanic or Latino'
    if derived_ethnicity == 2:
        derived_ethnicity == 'Not Hispanic or Latino'
    else: 
        derived_ethnicity = 'Joint'
                                  
    derived_race_dict = {1:'American Indian or Alaska Native', 2: 'Asian', 3: 'Black or African American',
                         4: 'Native Hawaiian or Other Pacific Islander', 5: 'White', 6: '2 or more minority races',
                         'Joint': 7}
    derived_race = derived_race_dict.get(derived_race)
    
    if derived_sex: 
        derived_sex = 'Male'
    if derived_sex == 2:
        derived_sex == 'Female'
    else: 
        derived_sex = 'Joint'
                                  
    loan_to_value_ratio = (loan_amount - down_payment)/loan_amount
                                  
    loan_term = 12 * loan_term
    
    debt_to_income_ratio = monthly_debt/income
                                  
    # Put DIR in bin
    if debt_to_income_ratio < 20:
        debt_to_income_ratio = 15
    if debt_to_income_ratio in range(20,30): 
       debt_to_income_ratio  = 25
    elif debt_to_income_ratio in range(30,36): 
        debt_to_income_ratio = 33
    elif debt_to_income_ratio in range(50,60):
        debt_to_income_ratio = 55
    else:
        debt_to_income_ratio = debt_to_income_ratio
    
    #hardcode age for now, will troubleshoot OrdinalEncoder later
#     applicant_age = np.int64(3.0)
    X_user_raw = np.array([census_tract, conforming_loan_limit, derived_dwelling_category, derived_ethnicity,
                      derived_race, derived_sex, loan_type, loan_amount, loan_to_value_ratio, loan_term,
                      construction_method,income, debt_to_income_ratio, applicant_age])
    
#     # Create dataframe for user
    columns = ['census_tract', 'conforming_loan_limit', 'derived_dwelling_category', 
                    'derived_ethnicity','derived_race', 'derived_sex', 'loan_type', 'loan_amount', 
                   'loan_to_value_ratio', 'loan_term', 'income', 'debt_to_income_ratio', 'applicant_age']
    X_user = pd.DataFrame(X_user_raw, columns= columns)
    # Not working - Ordinal encoding for age
#     enc_age_arr = enc.transform(X_user[['applicant_age']].to_numpy())
#     X_user['applicant_age'] = enc_age_arr
    
#     Get dummies
    categorical_cols = ['census_tract', 'conforming_loan_limit', 'derived_dwelling_category', 'derived_ethnicity', 
                        'derived_race', 'derived_sex', 'loan_type', 'construction_method']
                          
    X_user = pd.get_dummies(X_user, columns = categorical_cols)
    dummies_to_drop = ['derived_sex_Female', 'conforming_loan_limit_NC', 'construction_method_2', 
                       'derived_dwelling_category_Single Family (1-4 Units):Manufactured', 
                       'derived_ethnicity_Not Hispanic or Latino']
    X_user = X_user.drop(dummies_to_drop, axis=1)
    
    
    
    # predict on the new data
    model_lst = [boa, wells, chase, USB, LD, fair, cal]
    prob_lst = []
    for model in model_lst: 
        y_pred = model.predict(X_user)
        prob_lst.append(y_pred)
    
    return prob_lst
    
    
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
    return y_pred

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
