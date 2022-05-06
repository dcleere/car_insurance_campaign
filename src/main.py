# main executable for scoring new data
from flask import Flask, request, jsonify
from flask.logging import create_logger
from flask_restful import Resource, Api
import logging

import pandas as pd
import pickle
import numpy as np
from exploratory import env, transforms
#import env, transforms

app = Flask(__name__)
api = Api(app)
LOG = create_logger(app)
LOG.setLevel(logging.INFO)

# define cols
cols = ['Age', 'Job', 'Marital', 'Education', 'Default', 'Balance',
       'HHInsurance', 'CarLoan', 'LastContactDay', 'LastContactMonth',
       'NoOfContacts', 'DaysPassed', 'PrevAttempts', 'Outcome', 'CallStart',
       'CallEnd', 'CarInsurance', 'EducationEncoded']

class CarInsuranceClassifier(Resource):
    def post(self):
        """Performs prediction
        """
        try:
            with open(env.model_path+'best_xgb_cv.pkl', 'rb') as f:
                model = pickle.load(f)
        except:
            LOG.info("JSON payload: %s json_payload")
            return "Model not loaded"
        
        # get data
        json_payload = request.json
        LOG.info("JSON payload: %s json_payload")
        
        # convert to dataframe
        df_to_score = pd.DataFrame(json_payload, columns = cols)
        LOG.info("json coverted to dataframe")

        # impute missing
        df_to_score_imp = transforms.impute_vars(df_to_score)
        LOG.info("dataframe imputed for missing values")

        # create base table for scoring
        df = transforms.get_features(df_to_score_imp)
        LOG.info("dataframe transformed to model features")

        # convert to array as expected by the model
        ar_to_score = df.to_numpy()
        LOG.info("dataframe transformed to model features")

        # make prediction
        prediction = list(model.predict(ar_to_score))
        return jsonify({'prediction': prediction})

api.add_resource(CarInsuranceClassifier,"/predictions")  

if __name__ == "__main__":
    app.run(port=5000, debug=True, use_reloader=False)