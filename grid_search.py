import pprint as pp

from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import configurations as conf
from utils import debug_print

def hw3_grid_search(model_params_list):
	"""
	Grid search for Homework 3
	:param model_params_list: list of (models, parameters_ranges::dict)
	:return: a GridSearchCV instance
	"""

	labeled_models = [(name, model) for (name, model, _) in model_params_list]

	pipeline = Pipeline(labeled_models)

	parameters = dict(
		[(name + "__" + key, value)
		 for i,(name, _, params) in enumerate(model_params_list)
		 for key,value in params.items()]
	)

	grid_search = GridSearchCV(pipeline,
							   parameters,
							   scoring=metrics.make_scorer(metrics.matthews_corrcoef),
							   cv=conf.GRID_SEARCH_CV_PARAMS["cv"],
							   n_jobs=conf.GRID_SEARCH_CV_PARAMS["n_jobs"],
							   verbose=1)

	return grid_search

def print_grid_search_summary(grid_search):
	## debug_print results for each combination of parameters.
	number_of_candidates = len(grid_search.cv_results_['params'])
	debug_print("Results:")
	for i in range(number_of_candidates):
		debug_print(i, 'params - %s; mean - %0.3f; std - %0.3f' %
					(grid_search.cv_results_['params'][i],
					 grid_search.cv_results_['mean_test_score'][i],
					 grid_search.cv_results_['std_test_score'][i]))

	debug_print()
	debug_print("Best Estimator:")
	pp.pprint(grid_search.best_estimator_)
	debug_print()
	debug_print("Best Parameters:")
	pp.pprint(grid_search.best_params_)
	debug_print()
	debug_print("Used Scorer Function:")
	pp.pprint(grid_search.scorer_)
	debug_print()
	debug_print("Number of Folds:")
	pp.pprint(grid_search.n_splits_)
	debug_print()