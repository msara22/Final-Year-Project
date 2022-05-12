from msilib.schema import RadioButton
from flask_wtf import FlaskForm
from wtforms import IntegerField, SubmitField, RadioField
from wtforms.validators import InputRequired, NumberRange

class UserInput(FlaskForm):
    hour = IntegerField("Hour",validators=[InputRequired(),NumberRange(min=1,max=24)])
    day = IntegerField("Day",validators=[InputRequired(),NumberRange(min=1,max=30)])
    month = IntegerField("Month",validators=[InputRequired(),NumberRange(min=1,max=12)])
    year = IntegerField("Year",validators=[InputRequired(),NumberRange(min=2016,max=2022)])
    predict_probability = SubmitField("Predict Probability")
    choice = RadioField('choice',validators=[InputRequired()], coerce=int, choices=[(1, 'Daily'), (2, 'Weekly'), (3, "Monthly")],)
    

