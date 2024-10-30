import pandas as pd
import unittest
import warnings

link = 'https://2499737.youcanlearnit.net/tabulardata.html'

test_data = pd.read_html(link, match='department', header=0)

test_data = test_data[0]

test_data = test_data.drop('Unnamed: 10', axis=1)

def predictor_outcome_split(data_name):
    if not isinstance(data_name, pd.DataFrame):
       raise TypeError('The data_name argument must be a pandas data frame!')
    try:
        predictors = data_name.drop('separatedNY', axis=1)
        outcome = data_name['separatedNY']
        return predictors, outcome
    except KeyError:
        for col in data_name.columns:
            if data_name.columns.str.contains('[Ss]eparate').any():
                warnings.warn(f'Found a column that looks like the outcome variable: {col}')
                warnings.warn('Please confirm that this is the correct column before modeling!')
                data_name.rename(columns={col: 'separatedNY'}, inplace=True)
                predictors = data_name.drop('separatedNY', axis=1)
                outcome = data_name['separatedNY']
                return predictors, outcome
    raise KeyError('No separated outcome column detected!')
    
bad_columns = {'outcome': [1, 2, 3, 4, 5], 'age': [5, 4, 3, 2, 1]} 

bad_df = pd.DataFrame(data=bad_columns)

predictor_outcome_split(bad_df)

class TestPreparation(unittest.TestCase):
    def test_predictor_outcome_split(self):
       """Good Test"""
       test_data = pd.read_html(link, match='department', header=0)
       test_data = test_data[0]
       test_data = test_data.drop('Unnamed: 10', axis=1)
       predictors, outcome = predictor_outcome_split(test_data)
       self.assertIsInstance(predictors, pd.DataFrame)
       self.assertIsInstance(outcome, pd.Series)
    
    def test_bad_data(self):
       """Bad Test"""
       bad_columns = {'separated_ny': [1, 2, 3, 4, 5], 'age': [5, 4, 3, 2, 1]} 
       bad_df = pd.DataFrame(data=bad_columns)
       predictors, outcome = predictor_outcome_split(bad_df)
       self.assertIsInstance(predictors, pd.DataFrame)
       self.assertIsInstance(outcome, pd.Series)
    
    def test_worse_data_wrong(self):
        """Testing if worst data is returns an error"""
        worst_columns = {'outcome': [1, 2, 3, 4, 5], 'age': [15, 14, 13, 12, 11]}
        worst_df = pd.DataFrame(data=worst_columns)
        self.assertRaises(KeyError, predictor_outcome_split, (worst_df)) 
    
          
if __name__ == '__main__':
  unittest.main(verbosity=3)

  
