import warnings
from asl_data import SinglesData
import numpy as np


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for i in range(0,len(test_set.get_all_Xlengths())):
        X = test_set.get_item_sequences(i)
        length = test_set.get_item_Xlengths(i)

        probs = {}

        max_prob = -np.inf
        max_word = None

        for word, model in models.items():
            print(model.score(X,length))
            try:
                prob = model.score(X, length)
                probs[word] = prob

                if prob > max_prob:
                    max_word = word

            except ValueError:
                # if not work set default to zero log likelihood
                probs[word] = -np.inf

        probabilities.append(probs)
        guesses.append(max_word)

    return probabilities, guesses
