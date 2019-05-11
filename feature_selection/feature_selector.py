import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel
import shap


class FeatureSelector:

    def __init__(self, model, train_x, test_x, train_y, test_y, feature_names):
        '''
        This class should be initialized with a scikit-learn fitted model
        and four dataframes (or vectors) for training and testing features and labels
        '''
        
        print("Initiating FeatureSelector...")
        self.model = model
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y
        self.feature_names = feature_names
        print("Done.")

        print("Creating interpretator for model")
        self.interpreter = Interpretation(self.test_x, feature_names = self.test_x.columns)
        print("Done.")


        print("Building SHAP explainer for Tree Model")
        self.explainer = shap.TreeExplainer(self.model)
        print("Done.")

        print("Computing SHAP values for first 100 test samples...")
        self.shap_values = self.explainer.shap_values(test_x[:100])
        print("Done.")

        shap.initjs()
    
    def featureImportance(self):
        '''
        Calculates the importance of each feature in training set
        and the value of this feature in resulting output
        Outputs a horizontal bar chart.
        X-axis show the feature importance.
        Y-axis display the name of corresponding feature.
        '''

        # Load model in Skater memory

        mem_model = InMemoryModel(self.model.predict_proba, examples = self.train_x)

        # Generate feature importance plots 

        f = plt.figure(figsize=(10,3))

        plots = self.interpreter.feature_importance.plot_feature_importance(mem_model, ascending = True, progressbar=False)
        figure = plots[0]
        figure.set_size_inches(15, 8)
        ax = plots[1]
        ax.set_title("Feature importance of Random Forest model, trained on COMPAS dataset")
        ax.plot()


    def partialDependence(self, feature_names, tick_labels_list = None):
        '''
        Calculates the partial dependence of one or several features
        with the output label prediction.
        feature_names should be a list containing at least one string value which is
        feature name from the data.
        tick_labels_list should be a list of strings that would replace default ticks
        on the plot.
        Outputs line plot.
        X-axis show the values of the corresponding feature.
        Y-axis show the magnitude of partial dependence.
        '''

        mem_model = InMemoryModel(self.model.predict_proba, examples = self.train_x, 
                         target_names=['Probability of no recidive', 'Probability of recidive'])
    
        axes_list = self.interpreter.partial_dependence.plot_partial_dependence(feature_names, mem_model, grid_resolution=25, 
                                                                   with_variance=True, figsize = (8, 4), progressbar=False)
        ax = axes_list[0][1]
    
        title = 'Dependence between ' + feature_names[0] + ' and predicted label'
       
        if(tick_labels_list != None):
            ax.set_xticklabels(tick_labels_list)

        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.plot()

    def shapSummaryPlot(self):
        '''
        Shows summary plot of SHAP calculated values
        for each feature in the dataset relative to provided model.

        Outputs a corresponding bar plot.
        '''
        shap.summary_plot(self.shap_values, 
                              self.train_x, 
                              plot_type="bar", 
                              class_names = ["Recidive", "No recidive"],
                              title = "SHAP values of RF model output trained on COMPAS data")


    
        

    