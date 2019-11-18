# Disaster-Response-Pipeline-Project

### Table of Contents

1.  File Descriptions
2.  Results
3.  Instructions

## File Descriptions
There are three main foleders:

1. data
  - disaster_categories.csv: dataset including all the categories
  - disaster_messages.csv: dataset including all the messages
  - etl.py: ETL pipeline scripts to read, clean, and save data into a database
  - InsertDatabaseName.db: output of the ETL pipeline, i.e. SQLite database containing messages and categories data
2. models
  - ml.py: machine learning pipeline scripts to train and export a classifier
  - model.pkl: output of the machine learning pipeline, i.e. a trained classifer

3. app
  - run.py: Flask file to run the web application
  - templates contains html file for the web applicatin

## Results
An ETL pipleline was built to read data from two csv files, clean data, and save data into a SQLite database.
A machine learning pipeline was developed to train a classifier to performs multi-output classification on the 36 categories in the dataset.
A Flask app was created to show data visualization and classify the message that user enters on the web page.

## Instructions:
Run the following commands in the project's root directory to set up your database and model.

1. To run ETL pipeline that cleans data and stores in database python data/etl.py data/disaster_messages.csv data/disaster_categories.csv data/InsertDatabaseName.db
2. To run ML pipeline that trains classifier and saves python models/ml.py data/DisasterResponse.db models/model.pkl
3. Run the following command in the app's directory to run your web app. python run.py
4. To run your app with python run.py command
5. Open another terminal and type env|grep WORK this will give you the spaceid (it will start with view*** and some characters after that)
6. Now open your browser window and type https://viewa7a4999b-3001.udacity-student-workspaces.com, replace the whole viewa7a4999b with your space id that you got in the step 2
