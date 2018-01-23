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

    def __init__(self, all_word_sequences, all_word_Xlengths, this_word,
                 n_constant=None, min_n_components=2, max_n_components=10, random_state=14, verbose=False):
        ModelSelector.__init__(self, all_word_sequences, all_word_Xlengths, this_word, n_constant, min_n_components,
                               max_n_components, random_state, verbose)

        # set up attributes
        self.scores = []
        self.best_score = np.inf
        self.best_model = None
        self.n_features = len(self.words['ALL'][0][0])  # calculate the number of features from the words

    def _score_bic(self, n, logl):
        """ helper function to compute the bic score
        Bayesian information criteria: BIC = -2 * logL + p * log(d)
        Where LogL is the logLikelihood computed by the model.
        p is the number of parameters of the model
            p = |transition_matrix free parameters| + |emmision parameters| + |init probs|  # degrees of freedom
            p = n(n-1)  + nd +nd + n-1          (where d = number of features)
            p = n^2 -n +2nd +n-1 = n^2 +2nd -1
        N is the number of examples
        Args:
            n(int): number of states
            logl(float): the log likelihood

        """
        d = len(self.lengths)
        p = n ** 2 + 2 * n * d - 1  # the number of free parameters

        return -2 * logl + p * math.log(d)

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        for n_states in range(self.min_n_components, self.max_n_components + 1):

            model = self.base_model(n_states)
            score = np.inf

            if model is not None:
                try:
                    loglikelihood = model.score(self.X, self.lengths)
                    score = self._score_bic(n_states, loglikelihood)
                    self.scores.append(score)
                except ValueError:
                    pass

            if score < self.best_score:
                self.best_score = score
                self.best_model = model

        return self.best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def __init__(self, all_word_sequences, all_word_Xlengths, this_word,
                 n_constant=None, min_n_components=2, max_n_components=10, random_state=14, verbose=False):
        ModelSelector.__init__(self, all_word_sequences, all_word_Xlengths, this_word, n_constant, min_n_components,
                               max_n_components, random_state, verbose)

        # set up attributes
        self.scores = []
        self.best_score = np.inf
        self.best_model = None

    def _score_dic(self, model):
        """ helper function to compute the DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))"""

        words = self.hwords  # internal reference to the dict

        # compute  log(P(X(i))
        ll = model.score(self.X, self.lengths)

        # compute 1/(M-1)SUM(log(P(X(all but i))
        other_lls = np.mean([model.score(words[w][0], words[w][1]) for w in words if w != self.this_word])

        # compute dic
        return ll - other_lls

    def select(self):
        """ select the best model for self.this_word based on DIC between models trained on max and min components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        for n_states in range(self.min_n_components, self.max_n_components + 1):

            model = self.base_model(n_states)
            score = np.inf

            if model is not None:
                try:
                    score = self._score_dic(model)
                    self.scores.append(score)
                except ValueError:
                    pass

            if score < self.best_score:
                self.best_score = score
                self.best_model = model

        return self.best_model


class SelectorCV(ModelSelector):
    """ select best model based on average log Likelihood of cross-validation folds
    """

    def __init__(self, all_word_sequences, all_word_Xlengths, this_word,
                 n_constant=None, min_n_components=2, max_n_components=10, random_state=14, verbose=False):
        ModelSelector.__init__(self, all_word_sequences, all_word_Xlengths, this_word, n_constant, min_n_components,
                               max_n_components, random_state, verbose)

        # set up attributes
        self.mean_scores = []
        self.std_scores = []
        self.model_sizes = []
        self.best_score = -np.inf
        self.best_model = None

    def hmm_model(self, num_states, X, lengths):
        """ Runs the HMM model """
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

        # determine the number of CV iterations.
        n_splits = min([len(self.sequences), 5])


        for n_states in range(self.min_n_components, self.max_n_components + 1):

            score = -np.inf  # start a score

            # if only one example can not do cv # this is a bad model but oh well.
            if n_splits == 1:
                model =self.base_model(n_states)

                if model is not None:
                    try:
                        score = model.score(self.X, self.lengths)

                        if score > self.best_score:
                            self.best_score = score
                            self.best_model = n_states

                    except ValueError:
                        pass
            else: # we can do CV.
                split_method = KFold(random_state=self.random_state, n_splits=n_splits)

                scores = []
                for train_idx, test_idx in split_method.split(self.sequences):

                    # partition the data into X_train and X_test
                    X_train, l_train = combine_sequences(train_idx, self.sequences)
                    X_test, l_test = combine_sequences(test_idx, self.sequences)

                    # train
                    model = self.hmm_model(n_states, X_train, l_train)

                    if model is not None:
                        try:
                            score = model.score(X_test, l_test)
                            scores.append(score)

                        except ValueError:
                            pass

                if len(scores) > 0:
                    score = np.mean(scores)

                    # stat for CV...
                    self.mean_scores.append(score)
                    self.std_scores.append(np.std(scores))
                    self.model_sizes.append(n_states)

                if score > self.best_score:
                    self.best_score = score
                    self.best_model = n_states

        # retrain the model on all the data.
        if self.best_model is not None:
            return self.hmm_model(self.best_model, self.X, self.lengths)

        # if complete failure
        return None
