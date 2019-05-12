## Fairness audit for Random Forest model

### Team
1. Kurochkin Andrew, [SanGreel]().
2. Oleh Misko, [Progern](https://github.com/Progern).
3. Oleh Onyshchak, [OlehOnyshchak](https://github.com/OlehOnyshchak).
4. Valerii Veseliak, [ValeriyVeseliak](https://github.com/ValeriyVeseliak).

---
### Data
##### Compas dataset 
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

