## Fairness audit for Random Forest model
A survey made in 2018 highlighted two main concerns about AI for nowadays: model fairness and AI singularity, - most of the respondents chose model fairness as the main potential problem of the whole industry. Data scientists are 1.5 times more likely to consider issues around algorithmic fairness dangerous than any upcoming singularity when computers become more intelligent than people, the most of any kind of developer. Many developers discussed systemic bias being built into algorithmic decision making and the danger of AI being used without the ability to inspect and reason about decision pathways. <br/>
There are plenty of existing solutions to inspect models. Many of them work fine for their purpose, some already aggregated different features inside themselves. One concern is that these systems are often overload and too generalized. Due to this fact, we decided to build a solution for fairness audit for Random Forest model.<br/>
    
  In this work we created a pipeline for complete machine learning process illumination:
1. Found data used for risk assessment instruments (RAIs), COMPAS dataset that consists of a bunch of different sensitive features (sex, race, age, etc)
2. Inspected bias in the dataset
3. Built a model for recidivism prediction
4. Evaluated bias in the model prediction
5. Analyzed feature importance
6. Showed interpretation for prediction for new observations, that hadn’t been shown to the model previously.

---
### Team
1. Kurochkin Andrew, [SanGreel]().
2. Oleh Misko, [Progern](https://github.com/Progern).
3. Oleh Onyshchak, [OlehOnyshchak](https://github.com/OlehOnyshchak).
4. Valerii Veseliak, [ValeriyVeseliak](https://github.com/ValeriyVeseliak).

---
### Data
##### COMPAS dataset 
https://github.com/propublica/compas-analysis <br/>
https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing <br/>
https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm <br/>

---
### Instruction to reproduce
1. Download [COMPAS dataset](https://github.com/propublica/compas-analysis/) into folder "data".
2. Run  in the root folder:
```
pip install -r requirements.txt
```

3. [Optional] You can find some EDA of COMPAS dataset in the file [0_compas_data_exploration.ipynb](https://github.com/SanGreel/noble-model-audit/blob/master/0_compas_data_exploration.ipynb).
4. Run data preprocessing and preparation in the file [1_compas_data_preparation.ipynb](https://github.com/SanGreel/noble-model-audit/blob/master/1_compas_data_preparation.ipynb).
5. Now You should train the model for audit from the file [2_recidivism_prediction_model_training.ipynb](https://github.com/SanGreel/noble-model-audit/blob/master/2_recidivism_prediction_model_training.ipynb).
6. Congrats, now You are ready to go through fairness audit of Random Forest. All pipeline is in the file [3_recidivism_prediction_model_audit.ipynb](https://github.com/SanGreel/noble-model-audit/blob/master/3_recidivism_prediction_model_audit.ipynb).

---
### Audit examples
![](https://raw.githubusercontent.com/SanGreel/noble-model-audit/master/audit_result/EDA_sex.png)<br/>
![](https://raw.githubusercontent.com/SanGreel/noble-model-audit/master/audit_result/EDA_race.png)<br/>
![](https://raw.githubusercontent.com/SanGreel/noble-model-audit/master/audit_result/fairness_plot.png)<br/>
![](https://raw.githubusercontent.com/SanGreel/noble-model-audit/master/audit_result/fairness_plot2.png)<br/>
![](https://raw.githubusercontent.com/SanGreel/noble-model-audit/master/audit_result/partial_dependence.png)<br/>
![](https://raw.githubusercontent.com/SanGreel/noble-model-audit/master/audit_result/feature_selection.png)<br/>
![](https://raw.githubusercontent.com/SanGreel/noble-model-audit/master/audit_result/forest_tree_interpretation_0.jpg)

