from flask import Flask, render_template, request
import pickle
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from openai import AzureOpenAI
import markdown

app = Flask(__name__)

# Load Models and Preprocessors
learning_pace_model = pickle.load(open("models_and_preprocessor/learning_pace.pkl", "rb"))
average_score_model = load_model('models_and_preprocessor/average_score_model.h5', compile=False)
average_score_model.compile(optimizer='adam', loss='mse')
average_score_preprocessor = joblib.load('models_and_preprocessor/preprocessor.pkl')
student_level_model = joblib.load('models_and_preprocessor/logistic_model.pkl')
label_encoders = joblib.load('models_and_preprocessor/label_encoders.pkl')

# Azure OpenAI Configuration
ENDPOINT = "https://mango-bush-0a9e12903.5.azurestaticapps.net/api/v1"
API_KEY = "f839c878-573c-4d2d-984d-0c70a8618775"

API_VERSION = "2024-02-01"
MODEL_NAME = "gpt-4o"

client = AzureOpenAI(
    azure_endpoint=ENDPOINT,
    api_key=API_KEY,
    api_version=API_VERSION,
)

# Preprocessing Functions
def preprocess_data_for_pace(user_input):
    relevant_data = {
        'Age': user_input['age'],
        'Gender': 0 if user_input['sex'] == 'M' else 1,
        'ParentalEducation': {
            'HighSchool': 0, 'SomeCollege': 1, 'AssociateDegree': 2, 'Bachelor': 3, 'Master': 4
        }.get(user_input['ParentalEducation'], 0),
        'Absences': user_input['Absences'],
        'Tutoring': 1 if user_input['Tutoring'] == 'Yes' else 0,
        'ParentalSupport': 1 if user_input['ParentalSupport'] == 'Yes' else 0,
        'Extracurricular': 1 if user_input['Extracurricular'] == 'Yes' else 0,
        'GPA': user_input['GPA']
    }
    return relevant_data

def predict_learning_pace(model, student_data):
    preprocessed_data = preprocess_data_for_pace(student_data)
    student_df = pd.DataFrame([preprocessed_data])
    prediction = model.predict(student_df)[0]  # 0 for slow, 1 for fast
    return "fast learner" if prediction == 1 else "slow learner"

def preprocess_data_for_score(user_input):
    relevant_data = {
        'Gender': 'Male' if user_input['sex'] == 'M' else 'Female',
        'EthnicGroup': user_input['EthnicGroup'],
        'ParentEduc': user_input['ParentalEducation'],
        'LunchType': user_input['LunchType'],
        'TestPrep': user_input['TestPrep'],
        'ParentMaritalStatus': user_input['ParentMaritalStatus'],
        'PracticeSport': user_input['PracticeSport'],
        'IsFirstChild': user_input['IsFirstChild'],
        'WklyStudyHours': user_input['WklyStudyHours'],
        'MathScore': user_input['MathScore'],
        'ReadingScore': user_input['ReadingScore'],
        'WritingScore': user_input['WritingScore']
    }
    user_df = pd.DataFrame([relevant_data])
    return average_score_preprocessor.transform(user_df)

def predict_average_score(preprocessed_input):
    prediction = average_score_model.predict(preprocessed_input)
    return prediction[0][0]

def preprocess_data_for_level(user_input):
    encoded_data = {}
    for feature in label_encoders:
        value = user_input.get(feature)
        if value in label_encoders[feature].classes_:
            encoded_data[feature] = label_encoders[feature].transform([value])[0]
        else:
            # Handle unseen labels by using the first class as default
            encoded_data[feature] = label_encoders[feature].transform([label_encoders[feature].classes_[0]])[0]
    return pd.DataFrame([encoded_data])

def predict_student_level(user_input, model, label_encoders):
    preprocessed_data = preprocess_data_for_level(user_input)
    prediction = model.predict(preprocessed_data)[0]
    return "above avg" if prediction == 1 else "below avg"

# Combined Generative Method
def generate_study_plan(user_input):
    learning_pace = predict_learning_pace(learning_pace_model, user_input)
    average_score = predict_average_score(preprocess_data_for_score(user_input))
    student_level = predict_student_level(user_input, student_level_model, label_encoders)

    MESSAGES = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"I need help creating a study plan. \
         My predicted learning pace is {learning_pace}, if learning pace is fast, student can be multi-tasked and assign to 4 or more hours per day. Else assign student to study less than 4 hours per day. \
         Average score is around {average_score:.1f}, average score is calculated from past exam scores. \
         And student level is {student_level}, such level is categorized above average or below average depend on prediction model."}, 
        {"role": "assistant", "content": "Sure, I'd be happy to help! What subjects or topics do you need to include in your study plan?"},
        {"role": "user", "content": "The plan should cover mathematics over the next week."},
        {"role": "assistant", "content": "Do you have any specific goals or exams for these subjects during this period?"},
        {"role": "user", "content": "Yes, I'm preparing for midterms and need to cover calculus in mathematics."}
    ]

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES
    )

    response_content = completion.choices[0].message.content
    return response_content, learning_pace, average_score, student_level

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Extract form data
    user_input = {
        # Student Level Data
        'sex': request.form.get('sex'),
        'age': int(request.form.get('age')),
        'address': request.form.get('address'),
        'famsize': request.form.get('famsize'),
        'Pstatus': request.form.get('Pstatus'),
        'Medu': int(request.form.get('Medu')),
        'Fedu': int(request.form.get('Fedu')),
        'Mjob': request.form.get('Mjob'),
        'Fjob': request.form.get('Fjob'),
        'reason': request.form.get('reason'),
        'guardian': request.form.get('guardian'),
        'traveltime': int(request.form.get('traveltime')),
        'studytime': int(request.form.get('studytime')),
        'failures': int(request.form.get('failures')),
        'schoolsup': request.form.get('schoolsup'),
        'famsup': request.form.get('famsup'),
        'paid': request.form.get('paid'),
        'activities': request.form.get('activities'),
        'nursery': request.form.get('nursery'),
        'higher': request.form.get('higher'),
        'internet': request.form.get('internet'),
        'romantic': request.form.get('romantic'),
        'famrel': int(request.form.get('famrel')),
        'freetime': int(request.form.get('freetime')),
        'goout': int(request.form.get('goout')),
        'Dalc': int(request.form.get('Dalc')),
        'Walc': int(request.form.get('Walc')),
        'health': int(request.form.get('health')),
        'absences': int(request.form.get('absences')),

        # Average Score Data
        'EthnicGroup': request.form.get('EthnicGroup'),
        'LunchType': request.form.get('LunchType'),
        'TestPrep': request.form.get('TestPrep'),
        'ParentMaritalStatus': request.form.get('ParentMaritalStatus'),
        'PracticeSport': request.form.get('PracticeSport'),
        'IsFirstChild': request.form.get('IsFirstChild'),
        'WklyStudyHours': request.form.get('WklyStudyHours'),
        'MathScore': int(request.form.get('MathScore')),
        'ReadingScore': int(request.form.get('ReadingScore')),
        'WritingScore': int(request.form.get('WritingScore')),

        # Learning Pace Data
        'ParentalEducation': request.form.get('ParentalEducation'),
        'Absences': int(request.form.get('Absences')),
        'Tutoring': request.form.get('Tutoring'),
        'ParentalSupport': request.form.get('ParentalSupport'),
        'Extracurricular': request.form.get('Extracurricular'),
        'GPA': float(request.form.get('GPA'))
    }

    # Generate Study Plan
    study_plan, learning_pace, average_score, student_level  = generate_study_plan(user_input)
    study_plan = markdown.markdown(study_plan)

    return render_template('result.html', learning_pace=learning_pace, average_score=average_score, student_level=student_level, study_plan=study_plan)

if __name__ == '__main__':
    app.run(debug=True)
