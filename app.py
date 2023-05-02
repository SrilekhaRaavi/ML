import numpy as np
import pandas as pd
import pickle
from flask import Flask,request,jsonify,render_template


app = Flask(__name__, template_folder='templates', static_folder='static')
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    ##For rendering result on HTML interface
    if request.method=='POST':
        features = [x for x in request.form.values()]
        source_dict = {'Bangalore': 0, 'Chennai': 1, 'Delhi': 2, 'Kolkata': 3, 'Mumbai': 4}
        destination_dict = {'Bangalore':0,'Delhi':2,'Kolkata': 3,'Hyderabad':5,'New Delhi':6,'Cochin':7}
        airline_dict = {'IndiGo': 3, 'Air India': 1, 'Jet Airways': 4, 'SpiceJet': 8, 'Multiple carriers': 6, 'GoAir': 2, 'Vistara': 10, 'Air Asia': 0, 'Vistara Premium economy': 11, 'Jet Airways Business': 5, 'Trujet': 9}
        source_value = features[0]
        dest_value = features[1]
        date_value = features[2]
        airline_value = features[3]
        
        stops_value = int(features[4])   #<----------

        a= pd.Series(source_value)
        source = a.map(source_dict).values[0]   #<----------
        b= pd.Series(dest_value)
        destination = b.map(destination_dict).values[0] #<---------
        c= pd.Series(airline_value)
        airline = c.map(airline_dict).values[0]   #<----------

        day = int(pd.to_datetime(date_value, format="%Y-%m-%d").day)    #<----------------
        month = int(pd.to_datetime(date_value, format="%Y-%m-%d").month)  #<---------

        hour = int(pd.to_datetime(date_value, format ="%Y-%m-%d").hour)
        minute = int(pd.to_datetime(date_value, format ="%Y-%m-%d").minute)

           
        if  source==destination:
            return render_template('index1.html',pred='Source and Destination City cannot be same. Please try again! ')
            
        else:
            pred_features = [np.array([day,month,stops_value,hour,minute,airline,source,destination])]
            prediction = model.predict(pred_features)

            if stops_value==0:
                output = round(prediction[0],0)

            else:
                output = round(prediction[0],0)-2000
            print(output)


            return render_template('index1.html',pred='The Flight Fare for the given date is:-  INR {}'.format(output))
    else:
        return render_template('index1.html')
if __name__=='__main__':
    app.run(debug=True)
