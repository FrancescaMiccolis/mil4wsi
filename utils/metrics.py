import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
import copy
from sklearn.metrics import roc_curve,accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    labels=labels.reshape(-1,num_classes)
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def computeMetrics(test_labels, test_predictions, num_classes=2, names=[], optimal=False):
    if test_predictions.shape[0]==0:
        return None,None,None
    if optimal:
        auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, num_classes, pos_label=1)
    else:
        thresholds_optimal = [0.5]*num_classes
    class_prediction_bag = copy.deepcopy(test_predictions)
    if num_classes>1:
        for i in range(num_classes):
                class_prediction_bag = copy.deepcopy(test_predictions[:, i])
                class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
                class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
                test_predictions[:, i] = class_prediction_bag
    else:
        probabilities=copy.deepcopy(test_predictions)
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
    test_labels = np.squeeze(test_labels)
    acc=accuracy_score(test_labels,test_predictions)
    auc=roc_auc_score(test_labels,test_predictions)
    f1=f1_score(test_labels,test_predictions)
    precision=precision_score(test_labels,test_predictions)
    recall=recall_score(test_labels,test_predictions)
    cm=confusion_matrix(test_labels,test_predictions)
    specificity=recall_score(test_labels,test_predictions,pos_label=0)
    plt.clf()
    sns.set(font_scale=2)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax,xticklabels=np.unique(test_labels),yticklabels=np.unique(test_labels),annot_kws={"size": 45},cbar=False)
     # Set labels and title
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    # Add F1-Score and AUC to the plot
    plt.figtext(0.5, -0.05, f'F1-Score: {f1} | AUC: {auc}', ha="center", fontsize=18, fontweight='bold')

    plt.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    # Close the figure after saving it to avoid memory issues
    plt.close(fig)
    image = Image.open(buffer)
    # bag_score = 0
    # for i in range(0, len(test_predictions)):
    #     bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score
    # avg_score = bag_score / len(test_predictions)
    return class_prediction_bag,probabilities,acc,auc,f1,precision,recall,cm,specificity,image,thresholds_optimal[0]