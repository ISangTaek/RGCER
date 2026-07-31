import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import mean_squared_error, r2_score


class ClsMetric():
    r"""The AUROC and AUPRC metrics for Classification.
    """
    def __init__(self):
        self.auroc_record = []
        self.auprc_record = []
        self.bs = []

    def update_fun(self, pred, gt):
        r"""
        """
        pred = pred.cpu().numpy()
        gt = gt.cpu().numpy()

        fpr, tpr, _ = roc_curve(gt, pred)
        fpr_sorted, tpr_sorted = zip(*sorted(zip(fpr, tpr)))
        auroc_err = np.array(auc(fpr_sorted, tpr_sorted))

        pre, rec, _ = precision_recall_curve(gt, pred)
        pre_sorted, rec_sorted = zip(*sorted(zip(pre, rec)))
        auprc_err = np.array(auc(pre_sorted, rec_sorted))

        self.auroc_record.append(auroc_err)
        self.auprc_record.append(auprc_err)
        self.bs.append(pred.shape[0])
        
    def score_fun(self):
        r"""
        """
        auroc_record = np.array(self.auroc_record)
        auprc_record = np.array(self.auprc_record)
        bs = np.array(self.bs)
        return [(auroc_record*bs).sum()/(bs.sum()),
                (auprc_record*bs).sum()/(bs.sum())]

    def reinit(self):
        r"""
        """
        self.auroc_record = []
        self.auprc_record = []
        self.bs = []


class RegMetric():
    r"""The RMSE and R2 metrics for Regression.
    """
    def __init__(self):
        self.rmse_record = []
        self.r2_record = []
        self.bs = []

    def update_fun(self, pred, gt):
        r"""
        """
        pred = pred.cpu().numpy()
        gt = gt.cpu().numpy()

        rmse_err = np.sqrt(mean_squared_error(gt, pred))
        r2_err = np.array(r2_score(gt, pred))
        
        self.rmse_record.append(rmse_err)
        self.r2_record.append(r2_err)
        self.bs.append(pred.shape[0])
        
    def score_fun(self):
        r"""
        """
        rmse_record = np.array(self.rmse_record)
        r2_record = np.array(self.r2_record)
        bs = np.array(self.bs)
        return [(rmse_record*bs).sum()/(bs.sum()),
                (r2_record*bs).sum()/(bs.sum())]

    def reinit(self):
        r"""
        """
        self.rmse_record = []
        self.r2_record = []
        self.bs = []
