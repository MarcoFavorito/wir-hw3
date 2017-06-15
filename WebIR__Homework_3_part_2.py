import pprint as pp

from nltk.corpus import stopwords
from sklearn import metrics
from sklearn import svm
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import configurations as conf
from utils import debug_print
from utils import stemming_tokenizer


def main():

	## Dataset containing Positive and neative sentences on Amazon products
	dataset_folder = conf.POS_NEG_DATASET_DIR
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


	vectorizer = TfidfVectorizer(strip_accents= None, preprocessor = None)
	classifier = svm.SVC()


	pipeline = Pipeline([
		('vect', vectorizer),
		('classifier', classifier),
	])

	# stopwords
	en_stopwords = stopwords.words('english')

	# generate ngrams
	min_start = 1
	max_start = 3
	min_end = 1
	max_end = 10
	ngrams = [(x,y) for x in range(min_start,max_start) for y in range(min_end,max_end) if x<y]

	parameters = {
		'vect__tokenizer': [None, stemming_tokenizer],
		'vect__stop_words': [None, en_stopwords],
		# 'vect__max_df': [1.0, 0.9, 0.8, 0.7],
		# 'vect__min_df': [1,2,3],
		'vect__analyzer':['word', 'char'],
		# 'vect__ngram_range': ngrams,
		'vect__smooth_idf': [True, False],
		# 'vect__binary': [False, True],
		# 'vect__use_idf' : [True, False],
		'vect__sublinear_tf': [True, False],
		# 'vect__norm' : ['l2', 'l1'],
		'classifier__C':[1.0],
		'classifier__kernel':['linear'],
		'classifier__degree':[3],
		'classifier__gamma':['auto'],
		'classifier__coef0':[0.0],
		'classifier__shrinking':[True],
		'classifier__probability':[False],
		'classifier__tol':[0.001],
		'classifier__cache_size':[200],
		'classifier__class_weight':[None],
		'classifier__verbose':[False],
		'classifier__max_iter':[-1],
		'classifier__decision_function_shape':[None],
		'classifier__random_state':[None]
	}

	## Create a Grid-Search-Cross-Validation object
	## to find in an automated fashion the best combination of parameters.
	grid_search = GridSearchCV(pipeline,
							   parameters,
							   scoring=metrics.make_scorer(metrics.matthews_corrcoef),
							   cv=conf.GRID_SEARCH_CV_PARAMS["cv"],
							   n_jobs=conf.GRID_SEARCH_CV_PARAMS["n_jobs"],
							   verbose=2)

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

	# Let's train the classifier that achieved the best performance,
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


if __name__ == '__main__':
	main()