import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, auc, roc_curve
from joblib import dump, load



def train_gmm(train_features, train_targets):
    pair_models = []
    for i in range(10):
        data = train_features[train_targets==i]
        bestBIC = np.infty
        bestModel = None
        for comp_num in range(1, 3, 2):
            model = GaussianMixture(n_components=comp_num,
                                    covariance_type='full',
                                    reg_covar=1e-3,
                                    max_iter=100, init_params='kmeans', n_init=1
                                    )

            model.fit(data)
            curBic = model.bic(data)
            if curBic < bestBIC:
                bestBIC = curBic
                bestModel = model

        pair_models.append(bestModel)

    print(pair_models)
    dump(pair_models, './gmm_models')
    return pair_models


def compute_confusion_mat(yt, yp, class_nb=10):
    yt = np.asarray(yt, dtype=np.int)
    yp = np.asarray(yp, dtype=np.int)

    assert yt.shape == yp.shape

    conf_max = np.zeros((class_nb, class_nb), dtype=np.int)

    for i in range(class_nb):
        yp_c = yp[np.where(yt==i)]
        for j in range(class_nb):
            pred_j = np.where(yp_c==j)[0]
            num = pred_j.shape[0]
            conf_max[i][j] = num

    return np.asarray(conf_max, dtype=np.float)


def compute_log_density(models, feature):
    log_density = np.ones((feature.shape[0], 10))
    for i, m in enumerate(models):
        # lh = np.sum(np.exp(m._estimate_weighted_log_prob(nn_layer_outputs)), axis=1)
        # Estimate the weighted log-probabilities, log P(X | Z) + log weights
        log_density_class = m.score_samples(feature)
        log_density[:, i] = log_density_class

    return log_density


# print(np.exp(test_log_density[:10]), test_normal_preds[:10])
# print('*'*20)
# print(np.exp(adv_log_density[:10]), test_adv_preds[:10])

def kl(P, Q):
    return np.sum(np.where(P != 0, P * np.log(P / (Q)), 0))

def basic_ada(density_pred_log, density_exclued_log, q_pred, output_exluded):
    id = np.argmax(density_exclued_log)
    max_density_log = density_exclued_log[id]

    P_log = np.array([density_pred_log, max_density_log])
    P = np.exp(P_log - np.max(P_log))
    P = P / np.sum(P)

    Q = np.array([q_pred, output_exluded[id]])
    Q = Q / np.sum(Q)
    return kl(P, Q)

def compute_score(log_density, outputs, conf_mat):
    score_list = []
    kl_list = []
    same = 0
    predictions = np.argmax(outputs, axis=1)

    max_kls = []
    p_preds = []
    second_largests = []
    for i in range(0, log_density.shape[0]):
        log_density_i = log_density[i,:]
        output_i = outputs[i, :]
        conf_prob = conf_mat[:,  predictions[i]].copy()
        conf_prob = np.delete(conf_prob,  predictions[i])


        test_density = np.exp(log_density_i - np.max(log_density_i))
        test_density = test_density/np.sum(test_density)
        second_largest = np.max(np.delete(test_density, predictions[i]))

        kl_list.append(kl(test_density, output_i))
        p_preds.append(test_density[predictions[i]])

        second_largests.append(second_largest)

        if np.argmax(log_density_i) == predictions[i]:
            same+=1

        p_pred_log = log_density_i[predictions[i]]
        test_density_exclued_log = np.delete(log_density_i, predictions[i])

        q_pred = output_i[predictions[i]]
        test_output_exclued = np.delete(output_i, predictions[i])

        max_kl = basic_ada(p_pred_log, test_density_exclued_log, q_pred, test_output_exclued)

        max_kls.append(max_kl)

        pcs = test_density_exclued_log.copy()
        # lower_bound = np.max(pcs) * 10e-1
        # pcs = np.where(pcs < lower_bound, lower_bound, pcs)
        pcs = np.exp(pcs - np.max(pcs))
        pcs = pcs / np.sum(pcs)

        score = 0
        # if i == 0:
        #     print('pcs', pcs)


        for j in range(output_i.shape[0]-1):
            P_log = np.array([p_pred_log, test_density_exclued_log[j]])
            #print('p', P)
            Q = np.array([q_pred, test_output_exclued[j]])
            #print('q', Q)

            P = np.exp(P_log - np.max(P_log))
            P = P/np.sum(P)
            Q = Q/np.sum(Q)
            #print('P:', P,'\n', 'Q:',Q)

            temp = pcs[j] * np.sum(np.where(P != 0, P * np.log(P / (Q)), 0))
            temp = temp /(conf_prob[j])
            score += temp

        score_list.append(score)

    print('same number', same)
    return np.array(score_list), np.asarray(kl_list), np.asarray(max_kls), np.array(p_preds), np.array(second_largests)

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


def main():
    prefix = 'robust'
    train_features = np.load('./layeroutputs/{}_train_data.npy'.format(prefix))
    train_targets = np.load('./layeroutputs/{}_train_label.npy'.format(prefix))
    print(train_features.shape)
    print(train_targets.shape)

    test_normal_features = np.load('./layeroutputs/{}_test_data.npy'.format(prefix))
    test_normal_outputs = np.load('./layeroutputs/{}_test_outputs.npy'.format(prefix))
    test_normal_preds = np.argmax(test_normal_outputs, axis=1)

    test_adv_features = np.load('./layeroutputs/{}_adv_data.npy'.format(prefix))
    test_adv_outputs = np.load('./layeroutputs/{}_adv_outputs.npy'.format(prefix))

    test_adv_preds = np.argmax(test_adv_outputs, axis=1)

    val_pred = np.load('./confusionProbMat/{}_val_y_pred.npy'.format(prefix))
    val_targets = np.load('./confusionProbMat/{}_val_y_true.npy'.format(prefix))

    pair_models = train_gmm(train_features, train_targets)
    # pair_models = load('./gmm_models')

    conf_max = compute_confusion_mat(val_targets, val_pred)
    conf_max = np.where(conf_max == 0., 1, conf_max)
    for i in range(conf_max.shape[0]):
        conf_max[i, :] = conf_max[i, :] / np.sum(conf_max[i, :])
    print(conf_max)

    # train_log_density = compute_log_density(pair_models, train_features)
    test_log_density = compute_log_density(pair_models, test_normal_features)
    adv_log_density = compute_log_density(pair_models, test_adv_features)
    print(test_log_density.shape)
    norm_score, norm_kl, normal_max_kls, normal_p_preds, second_normal = compute_score(test_log_density,
                                                                                       test_normal_outputs,
                                                                                       conf_max
                                                                                       )

    adv_score, adv_kl, adv_max_kls, adv_p_preds, second_adv = compute_score(adv_log_density,
                                                                            test_adv_outputs,
                                                                            conf_max
                                                                            )

    fpr, tpr, auc_score = compute_roc(norm_score, adv_score)
    res = {'fpr': fpr,
           'tpr': tpr,
           'auc': auc_score
           }
    np.save('./res/Cifar_AW_ADA_roc.npy', res)

    plt.figure()
    plt.hist(normal_p_preds, 100, density=True, facecolor='g', alpha=0.75, label='normal test')
    plt.hist(adv_p_preds, 100, density=True, facecolor='b', alpha=0.75, label='adv samples')
    plt.legend(prop={'size': 10})
    plt.savefig('./images/p_preds')


if __name__ == '__main__':
    main()



# fpr, tpr, auc_score = compute_roc(normal_max_kls, adv_max_kls)
#
# n, bins, patches = plt.hist(normal_max_kls, 50, density=True, facecolor='g', alpha=0.75, label='normal test')
#
# n, bins, patches = plt.hist(adv_max_kls, 50, density=True, facecolor='b', alpha=0.75, label='adv samples')
# plt.legend(prop={'size': 10})
# plt.savefig('./images/max_kl')
#
# plt.figure()
# plt.hist(normal_p_preds, 100, density=True, facecolor='g', alpha=0.75, label='normal test')
# plt.hist(adv_p_preds, 100, density=True, facecolor='b', alpha=0.75, label='adv samples')
# plt.legend(prop={'size': 10})
# plt.savefig('./images/p_preds')
#
# plt.figure()
# plt.hist(second_normal, 100, density=True, facecolor='g', alpha=0.75, label='normal test')
# plt.hist(second_adv, 100, density=True, facecolor='b', alpha=0.75, label='adv samples')
# plt.legend(prop={'size': 10})
# plt.savefig('./images/second')
# train_features = np.load('./layeroutputs/Resnet18_v2_train_data.npy')
# train_targets = np.load('./layeroutputs/Resnet18_v2_train_targets.npy')
# plt.figure()
# print(train_features.shape)
# for i in range(10):
#     x_c = train_features[train_targets==i, :]
#     print(x_c.shape)
#     t = x_c[:, 0]
#     print(t.shape)
#     plt.scatter(x_c[:, 0], x_c[:, 1],
#             s=10,
#             edgecolors=(0, 0, 0),
#             label = 'c_{}'.format(i)
#             )
#
# plt.legend(loc="upper left")
# plt.show()


# plt.figure()
# plt.scatter(np.log(x_reduced[:, 0]), np.log(x_reduced[:, 1]), s=40, c='gray', edgecolors=(0, 0, 0))
# plt.scatter(np.log(x_adv[:, 0]), np.log(x_adv[:, 1]),
#             s=20,
#             c='red',
#             edgecolors=(0, 0, 0),
#             label = 'adv'
#             )
#
# plt.scatter(np.log(x_norm[:, 0]), np.log(x_norm[:, 1]),
#             s=20,
#             c='green',
#             edgecolors=(0, 0, 0),
#             label = 'norm'
#             )
#
# plt.scatter(np.log(x_reduced[:500, 0]), np.log(x_reduced[:500, 1]), s=10, edgecolors='b',
#             facecolors='none', linewidths=2, label='train src class')
# plt.scatter(np.log(x_reduced[500:, 0]), np.log(x_reduced[500:, 1]), s=10, edgecolors='orange',
#             facecolors='none', linewidths=2, label='train target class')


# plt.xlabel('{} feature'.format(25))
# plt.ylabel('{} feature'.format(30))
# plt.legend(loc="upper left")
# plt.show()

# plt.figure()
# plt.scatter(x_reduced[:, 0], x_reduced[:, 1], s=40, c='gray', edgecolors=(0, 0, 0))
# plt.scatter(x_adv[:, 0], x_adv[:, 1],
#             s=20,
#             c='red',
#             edgecolors=(0, 0, 0),
#             label = 'adv'
#             )
#
# plt.scatter(x_norm[:, 0], x_norm[:, 1],
#             s=20,
#             c='green',
#             edgecolors=(0, 0, 0),
#             label = 'norm'
#             )
#
# plt.scatter(x_reduced[:500, 0], x_reduced[:500, 1], s=10, edgecolors='b',
#             facecolors='none', linewidths=2, label='train src class')
# plt.scatter(x_reduced[500:, 0], x_reduced[500:, 1], s=10, edgecolors='orange',
#             facecolors='none', linewidths=2, label='train target class')
#
#
# plt.xlabel('{} feature'.format(25))
# plt.ylabel('{} feature'.format(30))
# plt.legend(loc="upper left")
# plt.show()