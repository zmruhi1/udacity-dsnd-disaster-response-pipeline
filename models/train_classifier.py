import sys
import re
import pickle
import pandas as pd
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

nltk.download(['punkt', 'wordnet', 'omw-1.4'])

def load_data(database_filepath):
    '''
    input   : path to SQLite database
    output  : returns features, ground truth and label names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterMessages', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])

    return X, Y, category_names


def tokenize(text):
    '''
    input   : messages 
    output  : returns case normalized, lemmatized, and tokenized text 
    '''
    # creating tokens
    words = word_tokenize(re.sub(r"[^a-zA-Z0-9]", ' ', text.lower()))
    # stemming
    stemmed_words = [PorterStemmer().stem(w) for w in words]
    # lemmatizing
    lemmatizer = WordNetLemmatizer()
    lemmed_words = [lemmatizer.lemmatize(w, pos='n').strip() for w in stemmed_words]
    lemmed_words = [lemmatizer.lemmatize(w, pos='v').strip() for w in stemmed_words]
    
    return lemmed_words

    
def build_model():
    '''
    input   : none
    output  : returns GridSearchCV object performed on the pipeline 
    '''
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])

    parameters = {'clf__estimator__n_estimators': [20, 50],
                  'vect__max_df': (0.5, 0.75, 1.0),
                  'clf__estimator__min_samples_split': [2, 3, 5],
                 }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    input   : model, test features and ground truth, category names
    output  : prints classification report for each category 
    '''
    preds = model.predict(X_test)
    
    for i in range(len(category_names)):
        print(Y_test.columns[i], ':')
        print(classification_report(Y_test.iloc[:,i], preds[:,i]))


def save_model(model, model_filepath):
    '''
    input   : model 
    output  : saves the model to given path 
    '''
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model.best_estimator_, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()