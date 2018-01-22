import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

def get_sublist(a, indicies):
    """helper function to get a list"""
    return [a[i] for i in indicies]



class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def _score_bic(self, n_components, n_features, log_likelihood, n_points):
        """ helper function to compute the bic score """
        parameters = n_components ** 2 + 2*n_components*n_features-1

        return -2* log_likelihood + parameters*math.log(n_points)

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores

        best_bic = np.inf
        best_model = None
        for n_components in range( self.min_n_components, self.max_n_components+1):
            model = self.base_model(n_components)
            log_likelihood = model.score(self.X, self.lengths)
            bic = self._score_bic(n_components, self.n_features, log_likelihood, len(self.lengths))

            if bic<best_bic:
                best_model = model

        return best_model





class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        raise NotImplementedError


class SelectorCV(ModelSelector):
    """ select best model based on average log Likelihood of cross-validation folds

    """

    def hmm_model(self,num_states, X, lengths ):
        try:
            model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X, lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        n_splits = min([len(self.sequences),5])

        split_method = KFold(random_state=self.random_state, n_splits=n_splits)

        best_score = -np.inf
        best_model_n = None

        for n_states in range(self.min_n_components, self.max_n_components+1):
            scores = []
            for train_idx, test_idx in split_method.split(self.sequences):

                # create the data sets
                X_train = combine_sequences(train_idx, self.sequences)
                l_train = get_sublist(self.lengths,train_idx)

                X_test = combine_sequences(test_idx, self.sequences)
                l_test = get_sublist(self.lengths,test_idx)

                # train
                #model = self.hmm_model(n_states, X_train, l_train)
                model = self.base_model(n_states)

                if model is not None:
                    score = model.score(X_test, l_test)
                    scores.append(score)
            if len(scores)>0:
                score = np.mean(score)
                if score>best_score:
                    best_score = score
                    best_model_n = n_states

        if best_model_n is not None:
            return self.hmm_model(best_model_n, self.X, self.lengths)












