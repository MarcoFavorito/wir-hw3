# let the library to know how many jobs it has to allocate
GRID_SEARCH_CV_PARAMS = {
	"n_jobs" : 8,
	"cv" : 10
}

VERBOSE = True
VERBOSITY = 2

HAM_SPAM_DATASET_DIR = "./data/Ham_Spam_comments/"
HAM_SPAM_TRAINING_SET_DIR = HAM_SPAM_DATASET_DIR + "Training/"
HAM_SPAM_TEST_SET_DIR = HAM_SPAM_DATASET_DIR + "Test/"