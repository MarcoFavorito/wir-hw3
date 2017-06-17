import pprint as pp
from os import system

from nltk.corpus import stopwords
from sklearn import metrics
from sklearn import neighbors
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD

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

	# from https://stackoverflow.com/questions/22763224/nltk-stopword-list
	# >	"stopwords.words('english') returns a list of lowercase stop words.
	# 	It is quite likely that your source has capital letters in it
	# 	and is not matching for that reason.
	en_stopwords = stopwords.words('english')

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



	## Vectorization object
	# ----------------------------------------
	# @Marco:	We can use the "preprocessor" or "stop_words" argument
	# 			To pass some callback function
	#			as explained by Fazzone:
	#
	#			"You are also allowed to use tools offered by NLTK libraries:
	# 			for example, to perform stemming and to have a set of stopwords."
	# ----------------------------------------

	# Fazzone's version:
	vectorizer = TfidfVectorizer(strip_accents=None, preprocessor=None)

	# for Multinomial Naive Bayes, it seems to perform worse than Fazzone...
	# with stopwords:	avg / total       0.90      0.90      0.90       105
	# without:			avg / total       0.93      0.92      0.92       105
	# The grid search always select stop_words=None. See after in "parameters".

	## classifier
	# nbc = MultinomialNB()
	knnc = neighbors.KNeighborsClassifier()

	svd = TruncatedSVD()
	## With a Pipeline object we can assemble several steps
	## that can be cross-validated together while setting different parameters.

	pipeline = Pipeline([
		('vect', vectorizer),
		('svd', svd),
		('knnc', knnc),
		])


	## Setting parameters.
	## Dictionary in which:
	##  Keys are parameters of objects in the pipeline.
	##  Values are set of values to try for a particular parameter.
	parameters = {
		'vect__tokenizer': [stemming_tokenizer], # [None, stemming_tokenizer],
		# 'vect__analyzer': ['word', 'char'],
		# 'vect__norm': ['l1', 'l2'],
		# 'vect__analyzer': ['char'],
		'vect__stop_words': [en_stopwords], # [None, en_stopwords],
		'vect__max_df': [1.0],
		'vect__min_df': [14],
		# 'vect__binary': [True, False],
		# 'vect__use_idf': [True, False],
		'vect__smooth_idf': [True],
		'vect__sublinear_tf': [True], # [True, False],
		# 'vect__ngram_range': [(1, x) for x in range(1, 2)],
		'vect__ngram_range': [(1,2)],
		# 'vect__ngram_range': [(3, x) for x in range(3, 10)],
		# 'svd__n_components': ['mle'],
		# 'svd__svd_solver' : ['full'],
		'svd__n_components':[6],
		'svd__algorithm': ['randomized'],
		# 'svd__n_iter': [5, 10],
		'knnc__n_neighbors': [7],
		'knnc__weights': ['distance'],
		# 'knnc__metric': ['minkowski', 'euclidean'],
		# 'knnc__p': [1, 2, 3],
		# 'knnc__algorithm': ['ball_tree', 'kd_tree'],
		# 'knnc__leaf_size' : [7, 15, 30, 60, 120]
		}


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


	fitted_vectorizer = grid_search.best_estimator_.steps[0][1]
	print("vocabulary len:",len(fitted_vectorizer.vocabulary_))
	print("vocabulary terms:")
	for k,v in fitted_vectorizer.vocabulary_.items():
		print("\t",k, v)

	for x,yt,yp in zip(X_test, Y_test, Y_predicted):
		if yt!=yp:
			print(yp, yt, x)
			A = fitted_vectorizer.transform([x])
			vectorized_sentence = A[0] # this will have a lot of missing words from the original sentence
			inverse_transformed_sentence = fitted_vectorizer.inverse_transform(vectorized_sentence)
			# subscriptor "[0]" because the result is:
			# inverse_transformed_sentence=[array(['video', 'song', 'perri', 'kati perri', 'kati'], dtype='<U12')]
			reconstructed_sentence = " ".join(inverse_transformed_sentence[0])
			print("reconstruction: ", reconstructed_sentence)




# grid_search.best_estimator_.steps[0][1].inverse_transform([grid_search.best_estimator_.steps[0][1].transform(["attention we need 10 million views more for firework to reach 500m we have only 1 and half day left for katy s birtgday listen how it could be possible gt gt just open different tabs from different browser gt gt dont watch full video remember we dont have time on hand its time wasting view it only for 30 sec plz thumbs up and share"])[0]])


# if __name__ == '__main__':
main()
system('say Fatto')
