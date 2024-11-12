# Microsoft_Fabric_Hackthon

[Devpost](https://devpost.com/software/ai-educational-planner?ref_content=user-portfolio&ref_feature=in_progress)

## Required Packages

To run this project, ensure you have the following Python packages installed:

- `Python 3.12`
- `pandas` 
- `numpy` 
- `matplotlib` 
- `seaborn` 
- `joblib`
- `scikit-learn 1.5.2` for machine learning models and tools:
  - `LogisticRegression` 
  - `RandomForestClassifier` 
  - `accuracy_score`, `confusion_matrix`, and `classification_report`
  - `KMeans`
  - `train_test_split` and `cross_val_score` 
  - `StandardScaler` 
- `imbalanced-learn` for handling imbalanced datasets:
  - `SMOTE` 
- `tensorflow`

You can install all required packages using the following command:

```bash
pip install pandas numpy matplotlib seaborn joblib scikit-learn imbalanced-learn tensorflow
````
> **Note:** If you encounter any package version issues (such as attribute errors), please refer to `requirements.txt` for specific package versions.

## Directory Structure and File Descriptions

### combined_generative

- **models_and_preprocessor/**
  - `average_score_model.h5`: Pre-trained Neural Network model file for predicting a student's average score base on several features.
  - `label_encoders.pkl`: Feature transformation for the logistic model.
  - `learning_pace.pkl`: Model file used to predict a student's learning pace based on input features.
  - `logistic_model.pkl`: Pre-trained logistic model for predicting student's performance level.
  - `preprocessor.pkl`: Saved pre-processor for pre-trained NN model to eliminate redundant transformation of the data.

- **static/**: Contains static assets like CSS used in the web application interface.

- **templates/**: Holds HTML templates for rendering the web application UI.

- **annie_model.ipynb**: Jupyter notebook with experiments and model development on predicting learning pace. 

- **app.py**: The main application script that runs the Flask web server, handling requests and serving the front-end interface.

- **combined_model.ipynb**: Jupyter notebook combining all models into a unified pipeline, coordinating inputs across multiple models.

- **elaine_model.ipynb**: Jupyter notebook with experiments and model development on predicting student's grade. 

- **rosie_model.ipynb**: Jupyter notebook with experiments and model development on predicting student's performance level.

## How to Run the Web Page

To start the web application and interact with it to get your tailored study plan, follow these steps:

1. **Navigate to the `combined_generative` folder**:

   ```bash
   cd combined_generative
   ```
2. Run the Flask application:

   ```bash
   python app.py
   ```
3. **Open the web application**:

   Once the application is running, open your web browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000).

4. **Interact with the Web Page**:

   Use the web interface to input your data and find out your personalized study plan!

Now you can explore the application to see the personalized study plans generated based on your input.


   
