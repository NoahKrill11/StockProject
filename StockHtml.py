from flask import Flask, render_template, request, redirect, url_for
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pandas_datareader import data
from datetime import date
import matplotlib.pyplot as plt 


today = date.today()
start_date = '2010-01-01'
end_date = today.strftime("%Y-%m-%d")

app = Flask(__name__)
#button =0

@app.route('/', methods=('GET', 'POST'))
def index():
	if request.method == 'POST':
		ticker = request.form['title']
		ticker = ticker.upper()
		df = data.DataReader(ticker, 'yahoo', start_date, end_date)
		df = df[['Close']] 
		forecast_out = 30
		df['Prediction'] = df[['Close']].shift(-forecast_out)
		X = np.array(df.drop(['Prediction'],1))
		X = X[:-forecast_out]
		Y=np.array(df['Prediction'])
		Y=Y[:-forecast_out]
		x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=.2)
		lr = LinearRegression()
		lr.fit(x_train,y_train)
		lr_confidence = lr.score(x_test, y_test)
		print("lr confidence = ", lr_confidence)
		x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
		lr_prediction = lr.predict(x_forecast)
		
		return render_template('index.html',stock = lr_prediction)
	else:
		return render_template('index.html')


app.run("0.0.0.0" , debug=True)