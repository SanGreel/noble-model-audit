import pandas as pd
import seaborn as sns
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.plotting import Plot

class BiasDetection:
	def __init__(self, data_path):
		self.df = self.get_data(data_path)
		
	def get_data(self, path):
		df = pd.read_csv(path)
		df = df[df['c_charge_degree'] != 'O']
		df = df[df['is_recid'] != -1]
		df = df[abs(df['days_b_screening_arrest'])<=30]
		
		df = df.rename(columns={'id':'entity_id', 'is_recid':'label_value'})
		df.loc[df['score_text'] == 'Low', 'score'] = str(0.0)
		df.loc[df['score_text'] != 'Low', 'score'] = str(1.0)
		
		df = df[['entity_id', 'score', 'label_value', 'race', 'sex', 'age_cat']]
		return df
		
	def get_ratios(self, feature, target):
		ratios = {}
		df = self.df
		for x in df[feature].unique():
			labels = df[df[feature] == x][target]
			counts = labels.value_counts()
			ratios[x] = counts[1] / counts[0]
			
		return ratios

	def check_dataset_bias(self, threshold=0.2):
		df = self.df
		non_attr_cols = ['score', 'model_id', 'as_of_date', 'entity_id', 'rank_abs', 'rank_pct', 'id', 'label_value']
		attr_cols = list(df.columns[~df.columns.isin(non_attr_cols)])
		bias_summary = {}
		for col in attr_cols:
			ratios = self.get_ratios(col, 'label_value')
			vals = list(ratios.values())
			bias_summary[col] = {
				'ratios': ratios,
				'fair': (max(vals) - min(vals)) < threshold
			}
			
		is_fair = True
		for x in bias_summary.values():
			is_fair = is_fair and x['fair']
		
		return {
			'fair': is_fair,
			'details': bias_summary
		}
		
	def get_model_fairness(self, level='model'):
		g = Group()
		xtab, _ = g.get_crosstabs(self.df)
		
		b = Bias()
		majority_bdf = b.get_disparity_major_group(xtab, original_df=self.df, mask_significance=True)
		
		f = Fairness()
		fdf = f.get_group_value_fairness(majority_bdf)
		f_res = fdf
		if level == 'model':
			f_res = f.get_overall_fairness(fdf)
		elif level == 'attribute':
			f_res = f.get_group_attribute_fairness(fdf)
		
		return f_res
	
	def plot_fairness(self, is_absolute=True):
		fdf = self.get_model_fairness(level='value')
		
		aqp = Plot()
		fg = None
		if is_absolute:
			fg = aqp.plot_fairness_group_all(fdf, ncols=5, metrics = "all", show_figure=False)
		else:
			attr_cols = list(fdf['attribute_name'].unique())
			fg = aqp.plot_fairness_disparity_all(fdf, attributes=attr_cols, significance_alpha=0.05, min_group_size=0.01, show_figure=False)
		
		return fg
