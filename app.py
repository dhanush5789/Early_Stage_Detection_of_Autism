from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import AUC
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pickle
import pandas as pd



app = Flask(__name__)

dependencies = {
    'auc_roc': AUC
}


random = pickle.load(open('result_random.pkl','rb'))

decision = pickle.load(open('result_decision.pkl','rb'))



 

@app.route("/")
@app.route("/index")
def index():
	return render_template('index.html')
    
@app.route("/login")
def login():
	return render_template('login.html')   
    


@app.route("/chart")
def chart():
	return render_template('chart.html') 

@app.route("/performance")
def performance():
	return render_template('performance.html')  	

@app.route('/upload')
def upload():
    return render_template('upload.html')  
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index(pd.RangeIndex(start=0, stop=len(df)), inplace=True)
        return render_template("preview.html",df_view = df)	




@app.route('/prediction', methods = ['GET', 'POST'])
def prediction():
    return render_template('prediction.html')





@app.route('/predict',methods=["POST"])
def predict():
    if request.method == 'POST':
        notice = request.form['notice']
        concentrate = request.form['concentrate']
        easy = request.form['easy']
        switch = request.form['switch']
        read = request.form['read'] 
        listening = request.form['listening']
        difficult = request.form['difficult']
        categories = request.form['categories']
        face = request.form['face']
        people = request.form['people']
        age = request.form['age']
        gender = request.form['gender']
        ethnicity = request.form['ethnicity']
        jundice = request.form['jundice']
        austim = request.form['austim']
        contry_of_res = request.form['contry_of_res']
        used_app_before = request.form['used_app_before']
        age_desc = request.form['age_desc']
        relation = request.form['relation']
         
        
        model = request.form['model']
        
		# Clean the data by convert from unicode to float 
        
        sample_data = [notice,concentrate,easy,switch,read,listening,difficult,categories,face,people,age,gender,ethnicity,jundice,austim,contry_of_res,used_app_before,age_desc,relation]
        print(sample_data)
        # clean_data = [float(i) for i in sample_data]
        # int_feature = [x for x in sample_data]
        int_feature = [float(i) for i in sample_data]
        print(int_feature)
    

		# Reshape the Data as a Sample not Individual Features
        
        ex1 = np.array(int_feature).reshape(1,-1)
        print(ex1)
		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)

        # Reloading the Model
        if model == 'RandomForestClassifier':
           result_prediction = random.predict(ex1)
           
            
        elif model == 'decision':
          result_prediction = decision.predict(ex1)
           
           
        
        # if result_prediction == 1:
        #     result = 'Control  (no pancreatic disease)'
        # elif result_prediction == 2:
        #     result = 'Benign (benign hepatobiliary disease)'  
        # elif result_prediction == 3:
        #     result = 'PDAC (Pancreatic ductal adenocarcinoma)'  
          

    return render_template('prediction.html', prediction_text= result_prediction[0], model = model)
@app.route('/performances')
def performances():
	return render_template('performances.html')   
	

	
if __name__ =='__main__':
	app.run(debug = True)


	

	


