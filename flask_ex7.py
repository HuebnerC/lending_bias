
from flask import Flask, request
import pickle
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

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

# Home page with form on it to submit new data
@app.route('/')
def get_new_data():
    return '''
        <form action="/predict" method='POST'>
          Property Address:<br>
          <input type="text" name="address"> 
          <br><br>
          Is the loan amount greater than $548,250?<br>
          <input type="text" name="conforming_loan_limit"> 
          <br><br>
          1 - Manufactured home or 2 - site-built?:<br>
          <input type="text" name="derived_dwelling_category"> 
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
          Down payment amount:<br> 
          <input type="text" name="down_payment"> 
          <br><br>
          Income:<br>
          <input type="text" name="income">
          <br><br>
          Monthly debt - Include: Monthly rent or house payment, Monthly alimony or child support payments, 
          Student, auto, and other monthly loan payments, 
          Credit card monthly payments (use the minimum payment), Other debts
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
    length = float(request.form['length'])
    width = float(request.form['width'])
    X_n = np.array([[length, width]])
    
    # predict on the new data
    Y_pred = model.predict(X_n)

    # for plotting 
    X_0 = trainX[trainY == 0] # class 0
    X_1 = trainX[trainY == 1] # class 1
    X_2 = trainX[trainY == 2] # class 2
    
    # color-coding prediction 
    if Y_pred[0] == 0:
        cp = 'b'
    elif Y_pred[0] == 1:
        cp = 'r'
    else:
        cp = 'g'

    if plt:
        plt.clf() # clears the figure when browser back arrow used to enter new data

    plt.scatter(X_0[:, 0], X_0[:, 1], c='b', edgecolors='k', label = 'class 0')
    plt.scatter(X_1[:, 0], X_1[:, 1], c='r', edgecolors='k', label = 'class 1')
    plt.scatter(X_2[:, 0], X_2[:, 1], c='g', edgecolors='k', label = 'class 2')
    plt.scatter(X_n[:, 0], X_n[:, 1], c=cp, edgecolors='k', marker = 'd', \
        s=100, label = 'prediction')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('Prediction plotted with training data')
    plt.legend()
        
    image = BytesIO()
    plt.savefig(image)
    out = image.getvalue(), 200, {'Content-Type': 'image/png'}
    return out

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
