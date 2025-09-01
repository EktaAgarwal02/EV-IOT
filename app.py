from flask import Flask,render_template,request
import joblib 
import 

app = Flask(__name__)
@login_manager.user_loader
def load_user(user_id)
    return User.query .get(int(user_id))
df = pd.read_csv('EV_Predictive_Maintenance_Dataset_15min.csv')
#load the saved model and preprocessing components
model = joblib.load('xgboost.pkl')
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about') 
def about():
    return render_template('about.html')

@app.route('/login')
def login():
    return render_template('login.html')


if __name__ == "__main__":
    app.run(debug=True)
