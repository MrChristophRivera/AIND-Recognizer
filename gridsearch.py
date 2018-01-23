# This is to create the data set
import numpy as np
import pandas as pd

from tqdm import tqdm_notebook
from asl_data import AslDb, SinglesData
from my_model_selectors import SelectorBIC, SelectorCV, SelectorDIC
from my_recognizer import recognize


selectors =[SelectorBIC, SelectorCV, SelectorDIC]

features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']
features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']
features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']

custom_features1 = ['delta-rx2', 'delta-ry2', 'delta-lx2', 'delta-lx2']
custom_features2 = ['delta-polar-rr', 'delta-polar-rtheta', 'delta-polar-lr', 'delta-polar-ltheta']
custom_features3 = ['delta-polar-rr2','delta-polar-rtheta2', 'delta-polar-lr2', 'delta-polar-ltheta2']

features_custom = custom_features1 + custom_features2 + custom_features3

features_sets = [features_ground, features_norm, features_polar, features_delta, features_custom]


def normalize(df):
    """ Helper function to normalize the means"""
    return (df - df.mean(axis=0))/df.std(axis = 0)


def cartesian_to_polar(x, y):
    """ converts to polar
    Args:
        x(iterable): the x coordinates
        y(iterable): the y coordinates
    Returns:
        r, theta
    """
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(x, y)  # note that this is reversed

    return list(r), list(theta)


def delta(x):
    """ helper function to compute the diff and fill na values
    Args:
        x(pd.Series)
    Returns:
        diff
    """
    return x.diff().fillna(0)


def delta2(x):
    """ computes the second order delta (accelaration)
    Args:
        x(pd.Series)
    Returns:
        delta2_x
    """
    return x.diff().diff().fillna(0)


def create_asl():
    """ Creates an ASL db and returns. """

    asl = AslDb()

    # create the ground features
    asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
    asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
    asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
    asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

    # add the normalize features
    cols = ['right-x', 'right-y', 'left-x', 'left-y']
    norm_df = asl.df.groupby('speaker')[cols].apply(normalize).rename(dict(zip(cols, features_norm)), axis=1)

    # compute polar features
    r1, theta1 = cartesian_to_polar(asl.df['grnd-rx'], asl.df['grnd-ry'])
    r2, theta2 = cartesian_to_polar(asl.df['grnd-lx'], asl.df['grnd-ly'])
    polar_df = pd.DataFrame(dict(zip(features_polar, [r1, theta1, r2, theta2])))
    polar_df.index = asl.df.index

    # compute delta features
    cols = ['grnd-rx', 'grnd-ry', 'grnd-lx', 'grnd-ly']
    delta_df = asl.df.groupby('speaker')[cols].apply(delta).rename(dict(zip(cols, features_delta)), axis=1)

    cols1 = ['grnd-rx', 'grnd-ry', 'grnd-lx', 'grnd-ly']
    cols2 = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']

    # generate custom features
    cf1_df = asl.df[cols1].apply(delta2).rename(dict(zip(cols1, custom_features1)), axis=1)
    cf2_df = polar_df.apply(delta).rename(dict(zip(cols2, custom_features2)), axis=1)
    cf3_df = polar_df.apply(delta2).rename(dict(zip(cols2, custom_features3)), axis=1)

    # concatenate
    custom_df = pd.concat([cf1_df, cf2_df, cf3_df], axis=1)

    # add all of the features together
    asl.df = pd.concat([asl.df, norm_df, polar_df, delta_df, custom_df], axis=1)

    return asl


def calculate_error_rate(guesses: list, test_set: SinglesData):
    """ Print WER and sentence differences in tabular form

    :param guesses: list of test item answers, ordered
    :param test_set: SinglesData object
    :return:
        nothing returned, prints error report

    WER = (S+I+D)/N  but we have no insertions or deletions for isolated words so WER = S/N
    """
    S = 0
    N = len(test_set.wordlist)
    num_test_words = len(test_set.wordlist)
    if len(guesses) != num_test_words:
        print("Size of guesses must equal number of test words ({})!".format(num_test_words))
    for word_id in range(num_test_words):
        if guesses[word_id] != test_set.wordlist[word_id]:
            S += 1.0
    return S/N


def grid_search():
    """ does a grid search"""
    asl = create_asl()

    feature_set_names = ['ground', 'norm', 'polar', 'delta', 'custom']
    selector_names = ['BIC', 'CV', 'DIC']

    feature_names = []
    select_method = []
    error_rates = []

    for i in tqdm_notebook(range(5)):
        features = features_sets[i]
        feature_name = feature_set_names[i]

        training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
        sequences = training.get_all_sequences()
        Xlengths = training.get_all_Xlengths()

        test_set =asl.build_test(features)

        for j in tqdm_notebook(range(3)):
            model_selector = selectors[j]
            selector_name = selector_names[j]

            models = {}
            for word in training.words:
                model = model_selector(sequences, Xlengths, word,
                                       n_constant=3).select()
                models[word] = model

            _, guesses = recognize(models, test_set)
            error = calculate_error_rate(guesses, test_set)

            feature_names.append(feature_name)
            select_method.append(selector_name)
            error_rates.append(error)

    return pd.DataFrame({'Features': feature_names, 'Selector':select_method, 'ErrorRate': error_rates})

if __name__ =='__main__':
    res = grid_search()
    print(res)


