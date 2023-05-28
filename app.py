from flask import Flask, render_template, request
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('ipl_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    batting_team = request.form.get('batting_team')
    bowling_team = request.form.get('bowling_team')
    city = request.form.get('city')
    runs_left = float(request.form.get('runs_left'))
    balls_left = float(request.form.get('balls_left'))
    wickets_left = float(request.form.get('wickets_left'))
    current_run_rate = float(request.form.get('current_run_rate'))
    required_run_rate = float(request.form.get('required_run_rate'))
    target = float(request.form.get('target'))

    # Create a DataFrame with the input values
    data = [[batting_team, bowling_team, city, runs_left, balls_left, wickets_left,
             current_run_rate, required_run_rate, target]]
    columns = ['BattingTeam', 'BowlingTeam', 'City', 'runs_left', 'balls_left',
               'wickets_left', 'current_run_rate', 'required_run_rate', 'target']
    input_df = pd.DataFrame(data, columns=columns)

    team1 = batting_team
    team2 = bowling_team

    # Make the prediction using the loaded model
    prediction = model.predict_proba(input_df)

    return render_template('result.html',
                           team1=team1,
                           team2=team2,
                           probability1=int(prediction[0, 0] * 100),
                           probability2=int(prediction[0, 1] * 100))

if __name__ == '__main__':
    app.run(debug=False)
