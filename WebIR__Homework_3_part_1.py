import pprint as pp
from os import system

from nltk.corpus import stopwords
from sklearn import metrics
from sklearn import neighbors
from sklearn.datasets import load_files
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

import configurations as conf
from utils import debug_print, stemming_tokenizer, porter_stemming_tokenizer, lancaster_stemming_tokenizer


def main():
	## Dataset containing Positive and neative sentences on Amazon products
	dataset_folder = conf.HAM_SPAM_DATASET_DIR
	data_folder_training_set = dataset_folder + conf.TRAINING_DIR
	data_folder_test_set = dataset_folder + conf.TEST_DIR

	training_dataset = load_files(data_folder_training_set)
	test_dataset = load_files(data_folder_test_set)
	debug_print()
	debug_print("----------------------")
	debug_print(training_dataset.target_names)
	debug_print("----------------------")
	debug_print()


	# Load Training-Set
	X_train, _, Y_train, _ = train_test_split(training_dataset.data,
														training_dataset.target,
														test_size=0.0)
	target_names = training_dataset.target_names

	# Load Test-Set
	_, X_test, _, Y_test = train_test_split(test_dataset.data,
														test_dataset.target,
														train_size=0.0)

	target_names = training_dataset.target_names
	debug_print()
	debug_print("----------------------")
	debug_print("Creating Training Set and Test Set")
	debug_print()
	debug_print("Training Set Size")
	debug_print(Y_train.shape)
	debug_print()
	debug_print("Test Set Size")
	debug_print(Y_test.shape)
	debug_print()
	debug_print("Classes:")
	debug_print(target_names)
	debug_print("----------------------")

	vectorizer = TfidfVectorizer(strip_accents=None, preprocessor=None)
	svd = TruncatedSVD(n_iter=50)
	knnc = neighbors.KNeighborsClassifier(metric="minkowski")

	pipeline = Pipeline([
		('vect', vectorizer),
		('svd', svd), # comment this line if you want to check the first version
		('knnc', knnc),
		])

	## Setting parameters.
	## Dictionary in which:
	## Keys are parameters of objects in the pipeline.
	## Values are set of values to try for a particular parameter.

	# switch to first to second and viceversa
	# toggling alternatively these lines of code
	# parameters = first_version_params
	parameters = second_version_params


	## Create a Grid-Search-Cross-Validation object
	## to find in an automated fashion the best combination of parameters.
	grid_search = GridSearchCV(pipeline,
							   parameters,
							   scoring=metrics.make_scorer(metrics.matthews_corrcoef),
							   cv=conf.GRID_SEARCH_CV_PARAMS["cv"],
							   n_jobs=conf.GRID_SEARCH_CV_PARAMS["n_jobs"],
							   verbose=conf.VERBOSITY)

	## Start an exhaustive search to find the best combination of parameters
	## according to the selected scoring-function.
	debug_print()
	grid_search.fit(X_train, Y_train)
	debug_print()

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


	#Let's train the classifier that achieved the best performance,
	# considering the select scoring-function,
	# on the entire original TRAINING-Set
	Y_predicted = grid_search.predict(X_test)

	# Evaluate the performance of the classifier on the original Test-Set
	output_classification_report = metrics.classification_report(
										Y_test,
										Y_predicted,
										target_names=target_names)
	debug_print()
	debug_print("----------------------------------------------------")
	debug_print(output_classification_report)
	debug_print("----------------------------------------------------")
	debug_print()

	# Compute the confusion matrix
	confusion_matrix = metrics.confusion_matrix(Y_test, Y_predicted)
	debug_print()
	debug_print("Confusion Matrix: True-Classes X Predicted-Classes")
	debug_print(confusion_matrix)
	debug_print()

	debug_print("metrics.accuracy_score")
	debug_print(metrics.accuracy_score(Y_test, Y_predicted))

	debug_print("Matthews corr. coeff")
	debug_print(metrics.matthews_corrcoef(Y_test, Y_predicted))

	return

# Set of parameters for the first version of the classifier
first_version_params = {
	'vect__analyzer': ['word'],
	'vect__tokenizer': [stemming_tokenizer],
	'vect__stop_words': [stopwords.words('english')],
	'vect__ngram_range': [(1, 1)],
	'vect__max_df': [0.3],
	'vect__min_df': [14],
	'vect__binary': [False],
	'vect__use_idf': [True],
	'vect__smooth_idf': [True],
	'vect__sublinear_tf': [True],
	'vect__norm': ['l1'],
	'knnc__n_neighbors': [7],
	'knnc__weights': ['uniform'],
	'knnc__p': [2],

	# Uncomment the following entries in order to perform a full grid seach
	# with the ranges described in the report.
	# 'vect__analyzer': ['word', 'char'],
	# 'vect__tokenizer': [None, stemming_tokenizer, porter_stemming_tokenizer, lancaster_stemming_tokenizer],
	# 'vect__ngram_range': [(x, y) for x in range(1, 2) for y in range(1, 9) if x <= y],
	# 'vect__stop_words': [None, stopwords.words('english')],
	# 'vect__max_df': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
	# 'vect__min_df': range(1,21),
	# 'vect__binary': [True, False],
	# 'vect__use_idf': [True, False],
	# 'vect__smooth_idf': [True, False],
	# 'vect__sublinear_tf': [True, False],
	# 'vect__norm': ['l1', 'l2'],
	# 'svd__n_components':range(1,11),
	# 'knnc__n_neighbors': range(3, 10),
	# 'knnc__weights': ['uniform', 'distance'],
	# 'knnc__p': [1, 2],
}

# Set of parameters for the second version of the classifier.
second_version_params = {
	'vect__analyzer': ['char'],
	'vect__tokenizer': [stemming_tokenizer],
	'vect__stop_words': [stopwords.words('english')],
	'vect__ngram_range': [(1, 6)],
	'vect__max_df': [1.0],
	'vect__min_df': [1],
	'vect__binary': [False],
	'vect__use_idf': [True],
	'vect__smooth_idf': [True],
	'vect__sublinear_tf': [True],
	'vect__norm': ['l2'],
	'svd__n_components':[6],
	'knnc__n_neighbors': [3],
	'knnc__weights': ['distance'],
	'knnc__p': [3],

	# Uncomment the following entries in order to perform a full grid seach
	# with the ranges described in the report.
	# 'vect__analyzer': ['word', 'char'],
	# 'vect__tokenizer': [None, stemming_tokenizer, porter_stemming_tokenizer, lancaster_stemming_tokenizer],
	# 'vect__ngram_range': [(x, y) for x in range(1, 2) for y in range(1, 9) if x <= y],
	# 'vect__stop_words': [None, stopwords.words('english')],
	# 'vect__max_df': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
	# 'vect__min_df': range(1,21),
	# 'vect__binary': [True, False],
	# 'vect__use_idf': [True, False],
	# 'vect__smooth_idf': [True, False],
	# 'vect__sublinear_tf': [True, False],
	# 'vect__norm': ['l1', 'l2'],
	# 'svd__n_components':range(1,11),
	# 'knnc__n_neighbors': range(3, 10),
	# 'knnc__weights': ['uniform', 'distance'],
	# 'knnc__p': [1, 2, 3],
}





if __name__ == '__main__':
	main()

