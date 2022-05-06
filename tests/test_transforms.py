#import unittest
import pandas as pd
import numpy as np
from transforms import calc_call_duration, oversample


#class TestCallDuration(unittest.TestCase):
def test_calc_call_duration():
    """
    The test checks if call calc is being computed correctly
    """
        
    source_data = {'CallStart': ['13:45:20', '12:06:10', '10:00:00'], 'CallEnd': ['13:46:20', '12:06:40', '11:00:00']}
    source_df = pd.DataFrame(data=source_data)
    result = calc_call_duration(source_df)
    
    expected_data = {'CallDuration': [1, 0.5]}
    expected_df = pd.DataFrame(data=expected_data)
    
    assert expected_df.equals(result), "Should be equal"
    
def test_oversample():
    """
    The test checks if the class is balanced once in the final output
    """
    
    source_data = {'CarInsurance': [1,0,0,0,0], 'Id': [1,2,3,4,5], 'Education': ['primary', 'secondary', 'secondary', 'secondary', 'teritary']}
    source_df = pd.DataFrame(data=source_data)
    result = oversample(source_df)

    expected_data = {'CarInsurance': [1,1,1,1,0,0,0,0], 'Id': [1,1,1,1,2,3,4,5], 
                     'Education': ['primary', 'primary', 'primary', 'primary', 'secondary', 'secondary', 'secondary', 'teritary']}
    expected_df = pd.DataFrame(data=expected_data)

    assert expected_df.equals(result), "Should be equal dataframes"