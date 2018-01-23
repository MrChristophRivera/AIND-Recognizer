import warnings
from asl_data import SinglesData


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

    for i in range(len(test_set)):
        X = test_set.get_item_sequences(i)
        length = test_set.get_item_Xlengths(i)

        probs = {}

        max_prob = 0
        max_word = None

        for word in models:
            try:
                prob = models[word].score(X, length)
                probs[word] = prob

                if prob > max_prob:
                    max_word = word

            except ValueError:
                # if not work set default to zero.
                probs[word] = 0

        probabilities.append(probs)
        guesses.apend(max_word)

    return probabilities, guesses
