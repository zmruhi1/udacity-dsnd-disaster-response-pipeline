# Disaster Response Pipeline Project

### Summary 
The project's final product is an ML web application that can categorize disaster-related messages. The training dataset provided by Figure Eight has 36 categories and 3 genres. The web page also provides further visual information on the dataset. The dataset is cleaned and stored in an SQLite database. The ML pipeline is built to access this database and consecutively trained and tuned using GridSearchCV. The final model is saved to be later accessed in order to generate predictions. 

### Codebase  
(The preparation notebooks are not needed for the final run. The codes are transferred to the following scripts)
1. data/process_data.py (ETL pipeline)
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. models/train_classifier.py (ML pipeline)
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. app/run.py
- Generates the web application and creates data visualizations 


### Instructions:
1. Install the requirements: `pip install -r requirements.txt`

2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Go to `app` directory: `cd app`

4. Run your web app: `python run.py`

5. Click the `PREVIEW` button to open the homepage
