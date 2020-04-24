import argparse
from sklearn.mixture import GaussianMixture
import numpy as np
import time
import matplotlib.pyplot as plt
from joblib import dump, load
from scipy.special import kl_div
from sklearn.metrics import roc_auc_score, auc, roc_curve
from scipy.stats import entropy

model_nm = ''
it = -1
eps = 10e1 * np.finfo(float).eps

def kl(P, Q):
    return np.sum(np.where(P != 0, P * np.log(P / (Q)), 0))

def compute_roc(probs_neg, probs_pos, plot=False):
    """
    TODO
    :param probs_neg:
    :param probs_pos:
    :param plot:
    :return:
    """
    probs = np.concatenate((probs_neg, probs_pos))
    labels = np.concatenate((np.zeros_like(probs_neg), np.ones_like(probs_pos)))
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = auc(fpr, tpr)
    print(auc_score)
    return fpr, tpr, auc_score

def trainGMMs(x, y, num_class):
    gmms = []
    for i in range(0, num_class):
        x_class = x[y == i]

        model = GaussianMixture(n_components=1,
                                covariance_type='full',
                                reg_covar=1e-3,
                                max_iter=100, init_params='kmeans', n_init=2
                                )

        model.fit(x_class)
        gmms.append(model)
    dump(gmms, './detectionModel/gmm')
    print('dump model to ./detectionModel/gmm')

def get_log_density_gmm(x, num_class):
    gmms = load('./detectionModel/gmm')
    print(x.shape)

    log_density_list = []
    for m in gmms:
        log_d = np.expand_dims(m.score_samples(x), axis=1)
        log_density_list.append(log_d)

    log_density = np.concatenate(log_density_list, axis=1)

    return log_density


def score_target_class_prediction(log_density, target_class):
    target_class_density = []
    for i in range(log_density.shape[0]):
        target_class_density.append(log_density[i, target_class[i]])

    target_class_density = np.array(target_class_density)
    print(target_class_density.shape)
    return target_class_density


def measure_score_target_class_prediction():
    ds = np.load('./features/{}_adv_ims_it_{}_out.npy'.format(model_nm, it), allow_pickle=True)
    print('load from ', './features/{}_adv_ims_it_{}_out.npy'.format(model_nm, it))
    data = ds.item().get('feature')
    label = ds.item().get('labels') #TODO: CHANGE LABLES
    softmax_out = ds.item().get('softmax')
    adv_log_density = get_log_density_gmm(data, 10)
    adv_prediction = np.argmax(softmax_out, axis=1)
    adv_score = score_target_class_prediction(adv_log_density, adv_prediction)

    ds = np.load('./features/{}_clean_ims_out.npy'.format(model_nm), allow_pickle=True)
    data = ds.item().get('feature')
    label = ds.item().get('labels')  # TODO: CHANGE LABLES
    clean_log_density = get_log_density_gmm(data, 10)
    softmax_out = ds.item().get('softmax')
    clean_prediction = np.argmax(softmax_out, axis=1)
    clean_score = score_target_class_prediction(clean_log_density, clean_prediction)

    _, _, auc = compute_roc(-clean_score, -adv_score)
    log_file = './{}_target_class_detection.txt'.format(model_nm)
    sample = open(log_file, 'a+')
    print('iteration: {}, AUC: {}'.format(it, auc), file=sample)


def score_full_kl(log_density, softmax_out):
    score = []
    for i in range(log_density.shape[0]):
        log_p = log_density[i]
        P = np.exp(log_p - np.max(log_p))
        P = P/np.sum(P)
        Q = softmax_out[i]

        Q = np.where(Q != 0, Q, eps)
        Q = Q/np.sum(Q)

        if 0 in Q:
            print('when compute kl: Q element has zero: Q is', Q)

        score.append(kl(P, Q))

    return np.array(score)



def measure_score_full_kl():
    ds = np.load('./features/{}_adv_ims_it_{}_out.npy'.format(model_nm, it), allow_pickle=True)
    print('load from ', './features/{}_adv_ims_it_{}_out.npy'.format(model_nm, it))
    data = ds.item().get('feature')
    label = ds.item().get('labels') #TODO: CHANGE LABLES
    softmax_out = ds.item().get('softmax')
    adv_log_density = get_log_density_gmm(data, 10)
    adv_prediction = np.argmax(softmax_out, axis=1)
    adv_score = score_full_kl(adv_log_density, softmax_out)

    ds = np.load('./features/{}_clean_ims_out.npy'.format(model_nm), allow_pickle=True)
    data = ds.item().get('feature')
    label = ds.item().get('labels')  # TODO: CHANGE LABLES
    clean_log_density = get_log_density_gmm(data, 10)
    softmax_out = ds.item().get('softmax')
    clean_prediction = np.argmax(softmax_out, axis=1)
    clean_score = score_full_kl(clean_log_density, softmax_out)

    _, _, auc = compute_roc(clean_score, adv_score)
    log_file = './{}_full_kl_detection.txt'.format(model_nm)
    sample = open(log_file, 'a+')
    print('iteration: {}, AUC: {}'.format(it, auc), file=sample)


def main():
    global model_nm
    global it
    it = args.it
    model_nm = args.model
    if it == 0:  # train GMM
        trainds = np.load('./features/MNIST_train_ims_out.npy', allow_pickle=True)
        print('load from ./features/MNIST_train_ims_out.npy')
        train_data = trainds.item().get('feature')
        train_label = trainds.item().get('label')
        trainGMMs(train_data, train_label, num_class=10)

    measure_score_target_class_prediction()
    measure_score_full_kl()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        default='MNIST')

    parser.add_argument('--it',
                        type=int,
                        default=1,
                        )
    args = parser.parse_args()
    main()
