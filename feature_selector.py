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

        self.model = model 

        self.train_x = train_x
        self.train_y = train_y

        self.test_x = test_x
        self.test_y = test_y

        self.feature_names = feature_names

        self.interpreter = Interpretation(test_x, feature_names = feature_names)

    
    def featureImportance(self):

        # Convert scikit-learn model to InMemory Skater model
        mem_model = InMemoryModel(self.model.predict_proba, examples = self.train_x) 

        # Generate plots for feature importance
        plot = self.interpreter.feature_importance.plot_feature_importance(mem_model, ascending = True, progressbar = False)

        ax = plot[1]
        ax.set_title("Feature importance of RF model, trained on COMPAS dataset")

    
        

    