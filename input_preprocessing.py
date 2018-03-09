# Input preprocessing
import pytest
from ssg_pymeans import Pymeans, InvalidInput
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

#' Convert raw input data to data frame
#'
#' @param raw Raw input
#' @param stage Fit or predict
#'
#' @return Cleaned data in tibble format
#' @export
#'
#' @examples

def input_preprocessing(raw,stage):
    try:
        data=pd.DataFrame(raw)
    except:
        print "Failed to convert data to data frame."

    try:
        ncols == 0
        if stage == "fit":
            ncols ==2
        else stage != "fit":
            ncols ==3
        len(data.columns) == ncols
    except:
        print 'Input data must have', ncols, 'columns'

    try:
         len(data.rows) >= 0
    except:
        print 'Input data cannot be empty.'


    try:
        is_numeric_dtype(df.iloc[:,0])
    except:
        'Input data must be numeric'


    try:
        is_numeric_dtype(df.iloc[:,1])
    except:
        'Input data must be numeric'

    # change colum names

    if stage == "fit":
        data.columns = ['a', 'b']
    else if stage == "predict":
        data.columns = ['a', 'b','cluster']

    # change cluster column to factor type

    if (stage != 'fit' and data.columns == 3):
        data['cluster'].astype(object)

    return data 
        



        
