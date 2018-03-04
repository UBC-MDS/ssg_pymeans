import os
import pandas as pd

default_data = pd.read_csv(os.path.join(os.path.dirname(__file__),
                    'sample_train.csv'))
