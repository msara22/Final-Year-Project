from flask import Flask, request, render_template, flash
from logging import FileHandler,WARNING
from graphviz import render
from sklearn.model_selection import PredefinedSplit
from form import UserInput
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import csv
from csv import writer

app = Flask(__name__, template_folder="templates")
file_handler = FileHandler('errorlog.txt')
file_handler.setLevel(WARNING)
app.config['SECRET_KEY'] = "dfewdfg89sd789dfgs89sdg8gs0g7dsfd8gdff7dsg890fd"
app.config['SQLAlCHEMY_DATABASE_URI'] ="sqlite:///C://Users//sarvinoz.toshpulotov//Desktop//App_flask//inputs.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = True
app.config['SQLAlCHEMY_DATABASE_URI'] ="sqlite:////C://Users//sarvinoz.toshpulotov//Desktop//App_flask//:memory:"


#Initialize db

db = SQLAlchemy(app)

class Inputs(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    hour = db.Column(db.Integer, nullable = False)
    day = db.Column(db.Integer, nullable = False)
    month = db.Column(db.Integer, nullable = False)
    year = db.Column(db.Integer, nullable = False)
    prediction = db.Column(db.String(50), nullable = False)
    output = db.Column(db.String, nullable = False)
    # date_created = db.Column(db.DateTime, default=datetime.utcnow )
## Create a function to return a string

    def __init__(self, hour, day, month, year, prediction, output):
        self.hour = hour
        self.day = day
        self.month = month
        self.year =year
        self.prediction = prediction
        self.output = output
        

    def __repr__(self):
        return '<Hour %r>' % self.hour
db.create_all()
db.session.query(Inputs).delete()

int_features = []
k = ''
@app.route("/",methods = ['GET','POST'])
def index():
    form = UserInput()
    if form.validate_on_submit():
        hour = form.hour.data
        day = form.day.data
        month = form.month.data
        year = form.year.data
        int_features = [hour,day,month,year]
        output = model(int_features,request.form.get('choice'))
        r_output = str(round(output,2))
        print(r_output)
        if (request.form.get('choice') == '1'):
             prediction = "Daily"
        elif(request.form.get('choice') == '2'): 
            prediction = 'Weekly'
        else:
            prediction = 'Monthly'  
        print(k)
        input = Inputs(hour, day, month, year, prediction, r_output)
        
        db.session.add(input)
        db.session.commit()

        entries = Inputs.query.order_by(Inputs.id)
        return render_template("index.html",form = form, output = r_output, entries =entries)
    return render_template("index.html",form = form)

# @app.route('/inputs')
# def entries(int_features:list):
#     hour = int_features[0]
#     day = int_features[1]
#     month = int_features[2]
#     year = int_features[3]

#     input = Inputs(hour, day, month,year)
    
#     db.session.add(input)
#     db.session.commit()
#     flash('Record was successfully added')
#     return render_template('inputs.html')

    

def history(l:list): 
    f = open('output.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(l)
    f.close()
    return  

def daily(l:list):
    print("daily")
    hour_from =l[0]
    day_from = l[1]
    month_from = l[2]
    year_from = l[3]
    s = 0
    for t in range(0, 23):
        if(hour_from % 24 ==0):
            if(day_from + 1 >=30):
                if(month_from==12):
                    year_from = year_from + 1
                    month_from = 1
                    day_from = 1
                    hour_from = 0
                else:
                    month_from = month_from + 1
                    day_from = 1
                    hour_from = 0
        else:
            hour_from = t + hour_from
            print(hour_from)
        s = s +  model(hour_from, day_from, month_from, year_from)
    return s
        

def weekly(l:list):
    print("weekly")
    hour_from =l[0]
    day_from = l[1]
    month_from = l[2]
    year_from = l[3]
    s = 0
    for d in range(0,7):
        day_from  = day_from + d
        s = s + daily([hour_from, day_from, month_from, year_from])
        print(s, " day ", day_from)
    return s
        
    
def monthly(l:list):
    print("monthly")
    hour_from =l[0]
    day_from = l[1]
    month_from = l[2]
    year_from = l[3]
   
    if(month_from + 1 > 12):
        month_from = 1
        day_from = (day_from +30)%30
        year_from = year_from +1
    else:
        month_from = month_from + 1
    return model([hour_from, day_from, month_from, year_from])
     

    
def model(l:list, choice):   
    df = pd.read_csv("C://Users//sarvinoz.toshpulotov//Desktop//App_flask//Dataset2.csv")
    loaded_model = pickle.load(open("C://Users//sarvinoz.toshpulotov//Desktop//App_flask//rand_model.pkl", 'rb'))  
    hour_from =l[0]
    day_from = l[1]
    month_from = l[2]
    year_from = l[3]
    # #for year
    # while((year_from in df["year"].values)== False):
    #     if(year_from<=2016):
    #         year_from = year_from + 1
    #     else:
    #         if(year_from<=2020):
    #             month_from = month_from - 1
    # else:
    #     df0 = df[(df.loc[:,'year'].values == year_from)]
    
    # For Month
    while((month_from in df["month"].values)== False):
        if(month_from>=12):
            month_from = month_from - 1
        else:
            if(month_from>=1):
                month_from = month_from + 1
    else:
        df1 = df[(df.loc[:,'month'].values == month_from)]
    
    #for Day
    while((day_from in df1["day"].values)== False):
        if(day_from>=30):
            day_from = day_from - 1
        else:
            if(day_from>=1):
                day_from = day_from + 1
    else:
        df2 = df1[(df1.loc[:,'day'].values == day_from)]

    #for year
    # while((year_from in df2["year"].values)== False):
    #     if(year_from<=2016):
    #         year_from = year_from + 1
    #     else:
    #         if(year_from<=2020):
    #             month_from = month_from - 1
    # else:
    #     df0 = df2[(df2.loc[:,'year'].values == year_from)]
    #for hour and year
    while((hour_from in df2["hour"].values)== False):
        if(hour_from>=24):
            hour_from = hour_from - 1
        else:
            if(month_from==1):
                hour_from = hour_from + 1
    else:
        df3 = df2[(df2.loc[:,'hour'].values == hour_from)]    
    df3 = df3.sample(n=1)
    x = df3.iloc[:,:-1]
    if(choice == '1'):
        print('choice1')
        x_test_weekly = df[(df['hour'].values == df3['hour'].values) &(df['day'].values == df3['day'].values) &(df['month'].values == df3['month'].values)].index
        final = df.iloc[x_test_weekly[0] - 24:x_test_weekly[0], :-1].values
        result = loaded_model.predict(final)
        return round(result.reshape(-1,1).sum(),2)
    if( choice == '2'):
        print('choice2')
        x_test_weekly = df[(df['hour'].values == df3['hour'].values) &(df['day'].values == df3['day'].values) &(df['month'].values == df3['month'].values)].index
        final = df.iloc[x_test_weekly[0] - 168:x_test_weekly[0], :-1].values
        result = loaded_model.predict(final)
        return round(result.reshape(-1,1).sum(),2)
    if(choice == '3'):
        print('choice3')
        x_test_weekly = df[(df['hour'].values == df3['hour'].values) &(df['day'].values == df3['day'].values) &(df['month'].values == df3['month'].values)].index
        final = df.iloc[x_test_weekly[0] - 720:x_test_weekly[0], :-1].values
        result = loaded_model.predict(final)
        print(result.reshape(-1,1).sum())
        return round(result.reshape(-1,1).sum(),2)



    
if __name__ == '__main__':
    app.debug = True
    app.run()