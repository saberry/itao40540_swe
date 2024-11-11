from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load('C:/Users/sberry5/Documents/teaching/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/data/lasso_model.joblib')

def prediction(model, input_data):
    input_data = [[input_data[var] for var in input_data.keys()]]
    prediction = model.predict(input_data)[0]
    return prediction
# Index(['age', 'numberPriorJobs', 'proportion401K', 'startingSalary',
       #'currentSalary', 'performance', 'monthsToSeparate', 'workDistance',
       #'department_1', 'department_2', 'department_3']
@app.route("/")

def index():
    return """
    <h1>Model API</hjson>

    <p>Use the /predict endpoint to make predictions</p>

    <p>You will need the following variables:</p>

    <ul>
        <li>age</li>
        <li>numberPriorJobs</li>
        <li>proportion401K</li>
        <li>startingSalary</li>
        <li>currentSalary</li>
        <li>performance</li>
        <li>monthsToSeparate</li>
        <li>workDistance</li>
        <li>department_1</li>
        <li>department_2</li>
        <li>department_3</li>
    </ul>

    """

@app.route("/predict_attrition", methods=['POST'])

def turnover_predict():
    content = request.json
    prediction_out = prediction(model, content)
    prediction_prob = str(prediction(model, content))
    prediction_out = 'Yes' if prediction_out > .5 else 'No'
    if prediction_out == 'Yes':
        prediction_message = "With a predicted probability of {}, the employee is likely to leave".format(prediction_prob)
    else: 
        prediction_message = "With a predicted probability of {}, the employee is unlikely to leave".format(prediction_prob)
    
    return jsonify({'prediction': prediction_out, 
                    'probability': prediction_prob, 
                    'message': prediction_message})

if __name__ == '__main__':
    app.run(debug=True)

"""
{
	"age":26, 
	"numberPriorJobs":2,
	"department_1":1,
    "department_2":0,
    "department_3":0,
	"proportion401K":65,
	"startingSalary":5000,
	"currentSalary":75000,
	"performance":2,
	"monthsToSeparate":0,
	"workDistance":10
} 
"""