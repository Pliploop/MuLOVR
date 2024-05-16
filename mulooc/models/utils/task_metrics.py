
from torchmetrics.functional import auroc, average_precision, accuracy, recall, precision, r2_score
from mir_eval.key import weighted_score
from mir_eval import tempo
import torch

## all metric functions return a dictionary with the metrics

def multilabel_metrics(logits, labels, n_classes, **kwargs):
    
    preds = torch.sigmoid(logits)
    labels = labels.long()
    aurocs = auroc(preds,labels,task = 'multilabel',num_labels = n_classes)
    ap_score = average_precision(preds,labels,task = 'multilabel',num_labels = n_classes)
    return {'auroc':aurocs,'ap':ap_score}

def mtat_top50_metrics(logits, labels, n_classes, **kwargs):
    return multilabel_metrics(logits, labels, n_classes)

def mtat_all_metrics(logits, labels, n_classes, **kwargs):
    return multilabel_metrics(logits, labels, n_classes)

def mtg_top50_metrics(logits, labels, n_classes, **kwargs):
    return multilabel_metrics(logits, labels, n_classes)

def mtg_genre_metrics(logits, labels, n_classes, **kwargs):
    return multilabel_metrics(logits, labels, n_classes)

def mtg_instr_metrics(logits, labels, n_classes, **kwargs):
    return multilabel_metrics(logits, labels, n_classes)

def mtg_mood_metrics(logits, labels, n_classes, **kwargs):
    return multilabel_metrics(logits, labels, n_classes)
    
    
def giantsteps_metrics(logits, labels, n_classes):
    
    idx2class = {0: 'Eb minor', 1: 'A major', 2: 'F minor', 3: 'D minor', 4: 'G minor', 5: 'C minor', 6: 'A minor', 7: 'B minor', 8: 'Db minor', 9: 'D major', 10: 'E minor', 11: 'Bb major', 12: 'Ab minor', 13: 'C major', 14: 'Db major', 15: 'Ab major', 16: 'E major', 17: 'G major', 18: 'B major', 19: 'Gb minor', 20: 'Gb major', 21: 'Bb minor', 22: 'F major', 23: 'Eb major'}
    
    
    preds = torch.softmax(logits,dim = 1)
    preds = torch.argmax(preds,dim = 1)
    batch_size = preds.size(0)
    # preds_names = [idx2class[pred] for pred in preds.cpu().numpy()]
    labels_idx = torch.argmax(labels,dim = 1)
    # labels_names = [idx2class[label] for label in labels_idx.cpu().numpy()]
    accuracy_ = accuracy(preds,labels_idx, task = 'multiclass', num_classes = n_classes)
    weighted_score_ = 0
    
    labels_names = [idx2class[label] for label in labels_idx.cpu().numpy()]
    preds_names = [idx2class[pred] for pred in preds.cpu().numpy()]
    
    for i in range(batch_size):
        weighted_score_ += weighted_score(reference_key = labels_names[i], estimated_key = preds_names[i])
    weighted_score_ = weighted_score_/batch_size
    
    return {'accuracy':accuracy_,'weighted_score':weighted_score_}


def nsynth_pitch_metrics(logits, labels, n_classes, **kwargs):
    return get_multiclass_metrics(logits, labels, n_classes, 'nsynth_pitch')

def nsynth_pitch_special_metrics(logits, labels, n_classes, **kwargs):
    return get_multiclass_metrics(logits, labels, n_classes, 'nsynth_pitch_special')

def nsynth_instr_family_metrics(logits, labels, n_classes):
    return get_multiclass_metrics(logits, labels, n_classes, 'nsynth_instr_family')

def gtzan_metrics(logits, labels, n_classes, **kwargs):
    return get_multiclass_metrics(logits, labels, n_classes, 'gtzan')

def vocalset_technique_metrics(logits, labels, n_classes):
    return get_multiclass_metrics(logits, labels, n_classes, 'vocalset_technique')

def vocalset_singer_metrics(logits, labels, n_classes, **kwargs):
    return get_multiclass_metrics(logits, labels, n_classes, 'vocalset_language')

def medleydb_metrics(logits, labels, n_classes, **kwargs):
    return get_multiclass_metrics(logits, labels, n_classes, 'medleydb')

def emomusic_metrics(logits, labels):
    
    global_r2 = r2_score(preds = logits, target = labels, multioutput = 'uniform_average')
    v_r2 = r2_score(preds = logits[:,1], target = labels[:,1])
    a_r2 = r2_score(preds = logits[:,0], target = labels[:,0])
    
    return {
        'r2_score':global_r2,
        'valence_r2_score':v_r2,
        'arousal_r2_score':a_r2
    }
    
def get_multiclass_metrics(logits, labels, n_classes, name):
    preds = torch.softmax(logits,dim = 1)
    preds = torch.argmax(preds,dim = 1)
    batch_size = preds.size(0)
    # preds_names = [idx2class[pred] for pred in preds.cpu().numpy()]
    labels_idx = torch.argmax(labels,dim = 1)
    # labels_names = [idx2class[label] for label in labels_idx.cpu().numpy()]
    accuracy_ = accuracy(preds,labels_idx, task = 'multiclass', num_classes = n_classes)
    precision_ = precision(preds,labels_idx, task = 'multiclass', num_classes = n_classes)
    recall_ = recall(preds,labels_idx, task = 'multiclass', num_classes = n_classes)
    
    return {'accuracy':accuracy_,'precision':precision_,'recall':recall_}

def gtzan_vs_all_tempo_metrics(logits, labels,n_classes):
    return get_tempo_metrics(logits, labels)

def hainsworth_vs_all_tempo_metrics(logits, labels,n_classes):
    return get_tempo_metrics(logits, labels)

def giantsteps_vs_all_tempo_metrics(logits, labels,n_classes):
    return get_tempo_metrics(logits, labels)

def acmm_vs_all_tempo_metrics(logits, labels,n_classes):
    return get_tempo_metrics(logits, labels)

def gtzan_tempo_metrics(logits, labels,n_classes):
    return get_tempo_metrics(logits, labels)

def hainsworth_tempo_metrics(logits, labels,n_classes):
    return get_tempo_metrics(logits, labels)

def giantsteps_tempo_metrics(logits, labels,n_classes):
    return get_tempo_metrics(logits, labels)

def acmmirum_tempo_metrics(logits, labels,n_classes):
    return get_tempo_metrics(logits, labels)


def get_tempo_metrics(logits, ground_truth, tol = 0.04):
    
    
    #all to cpu
    logits = logits.cpu()
    ground_truth = ground_truth.cpu()
    
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)

    # Get the estimated tempi by finding the tempo class with the highest probability
    estimated_tempi = torch.argmax(probs, dim=-1)
    gt_tempi = torch.argmax(ground_truth, dim=-1)

    # Compute the Acc1 metric
    acc1 = torch.abs(estimated_tempi - gt_tempi) / gt_tempi <= tol
    acc2 = torch.min(torch.abs(estimated_tempi.unsqueeze(-1) - gt_tempi.unsqueeze(-1) * torch.tensor([0.5, 1, 2, 3, 1/3])), dim=-1)[0] / gt_tempi <= tol


    # Compute the Acc2 metric

    # Reduce the metrics for the batch by computing the mean
    acc1 = acc1.float().mean()
    acc2 = acc2.float().mean()

    return {'Acc1': acc1, 'Acc2': acc2}