from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

model = joblib.load('C:/Users/sberry5/Documents/teaching/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/data/lasso_model.joblib')

def prediction(model, input_data):
    input_data = [[float(input_data[var]) for var in input_data.keys()]]
    print(input_data)
    prediction = model.predict(input_data)[0]
    
    return prediction

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/turnover_predict", methods=['POST'])
def turnover_predict():
    content = request.form.to_dict()
    
    prediction_out = prediction(model, content)
    prediction_prob = str(prediction(model, content))
    prediction_out = 'Yes' if prediction_out > .5 else 'No'
    if prediction_out == 'Yes':
        prediction_message = "With a predicted probability of {}, the employee is likely to leave!".format(prediction_prob)
    else: 
        prediction_message = "With a predicted probability of {}, the employee is unlikely to leave!".format(prediction_prob)
    
    output = jsonify({'prediction': prediction_out, 
                      'probability': prediction_prob, 
                      'message': prediction_message})
    
    return render_template('predict_attrition.html', prediction_message=prediction_message)

#if __name__ == '__main__':
#    app.run(debug=True)

# On a Windows machine, you can run the app by typing the following command in the terminal:
# set FLASK_APP=flask_form_app.py
# flask run
# Don't forget to cd to the directory where the file is located before running the commands
