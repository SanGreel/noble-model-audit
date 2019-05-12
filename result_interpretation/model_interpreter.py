from skater.model import InMemoryModel
from skater.core.explanations import Interpretation
from lime.lime_tabular import LimeTabularExplainer
from treeinterpreter import treeinterpreter as ti
import six
import pydot

from sklearn.tree import export_graphviz
from sklearn import tree
import pandas as pd
from matplotlib import pyplot as plt
    
class ModelInterpreter:
   
    def __init__(self, random_forest_model, x_train, y_train):
        self.rf_model = random_forest_model
        self.x_train = x_train
        self.y_train = y_train
        self.columns = list(x_train.columns)
        self.explainer = LimeTabularExplainer(
            x_train.values,
            feature_names = self.columns
        )
        self.model = InMemoryModel(self.rf_model.predict_proba, examples = self.x_train)
        self.interpreter = Interpretation(training_data=self.x_train,feature_names=self.columns,
                                          training_labels=self.y_train)
        
    def explain_instance(self, instance):
        explained = self.explainer.explain_instance(
            instance, self.rf_model.predict_proba, num_features=len(self.columns)
        )
        explained.show_in_notebook()
    
    def print_tabular_feature_importance(self):
        print(self.interpreter.feature_importance.feature_importance(self.model))
        
    def __calculate_variables_contribution(self, record):
        for i,row in record.iterrows():
            data_point = pd.DataFrame([row])
            data_point.set_axis(['value_variable'], inplace=True) # Once transposed, it will be the column name
            prediction, bias, contributions = ti.predict(self.rf_model, data_point)

            local_interpretation = data_point.append(
                pd.DataFrame([[round(c[1],3) for c in contributions[0]]], columns=data_point.columns.tolist(), index=['contribution_variable'])
            ).T.sort_values('contribution_variable', ascending=False)
        return local_interpretation


    def print_instance_tree_interpretation(self, record):
        print(self.__calculate_variables_contribution(record))
        
    def visualize_tree(self, estimators, columns , folder = 'audit_result', filename='forest_tree_interpretation'):

        result = []
        dotfile = six.StringIO()

        for i, tree_in_forest in enumerate(estimators):

            export_graphviz(tree_in_forest, \
                            out_file=folder+'/tree.dot',\
                            feature_names=columns,\
                            filled=True,\
                            rounded=True
                           )

            (graph,) = pydot.graph_from_dot_file(folder+'/tree.dot')
            name = folder + '/' + filename + '_' + str(i) +  '.png'
            graph.write_png(name)
            result.append(name)
            #os.system('dot -Tpng tree.dot -o tree.png')
            i +=1

        return result
    
    
