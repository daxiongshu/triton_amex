import numpy as np

def amex_metric_np(target: np.ndarray, preds: np.ndarray) -> float:
    indices = np.argsort(preds)[::-1]
    preds, target = preds[indices], target[indices]

    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_mask = cum_norm_weight <= 0.04
    d = np.sum(target[four_pct_mask]) / np.sum(target)

    weighted_target = target * weight
    lorentz = (weighted_target / weighted_target.sum()).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    n_pos = np.sum(target)
    n_neg = target.shape[0] - n_pos
    gini_max = 10 * n_neg * (n_pos + 20 * n_neg - 19) / (n_pos + 20 * n_neg)

    g = gini / gini_max
    return 0.5 * (g + d)

def amex(y_pred, y_true):
    return 'amex', amex_metric_np(y_pred,y_true)

# X_test = preprocess(X_test)
# trainl = cudf.read_csv(f'{PATH}/train_labels.csv')
# X_test = X_test.merge(trainl, on='customer_ID', how='left')
# X_test = X_test.sort_values('cid')
# X_test = X_test.reset_index(drop=True)
# y_test = X_test['target']
# not_used = get_not_used()
# X_test = X_test.drop(not_used,axis=1)