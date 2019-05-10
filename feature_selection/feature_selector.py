import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel

class FeatureSelector:
    
    def __init__(self, model, train_x, test_x, train_y, test_y, feature_names):
        '''
        This class should be initialized with a scikit-learn fitted model
        and four dataframes (or vectors) for training and testing features and labels
        '''

        featureImportance(model, train_x, test_x)

    
    def featureImportance(model, train_x, test_x):

        # Create interpreter for our testing data

        interpreter = Interpretation(test_x, feature_names = df.columns)

        # Load model in Skater memory

        mem_model = InMemoryModel(model.predict_proba, examples = train_x)

        # Generate feature importance plots 

        plots = interpreter.feature_importance.plot_feature_importance(mem_model, ascending = True, progressbar=False)
        ax = plots[1]
        ax.set_title("Feature importance of RF model, trained on COMPAS dataset")

        ax.show()
    
        

    