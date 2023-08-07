from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, f1_score
import pandas as pd

def calculate():
    gt_leision = []
    hypo_leision = []
    hypo_confidence = []

    with open("generation.csv", "r") as f:
        for line in f.readlines():
            fields = line.split("\t")
            hypo = fields[1]
            gts = fields[2]
            label = fields[3]
            if "low grade" in hypo:
                hypo_leision.append(0)
            elif "high grade" in hypo:
                hypo_leision.append(1)
            else:
                hypo_leision.append(2)

            if "low grade" in gts.lower():
                gt_leision.append(0)
                # gt_leision_reverse.append(1)
            elif "high grade" in gts.lower():
                gt_leision.append(1)
                # gt_leision_reverse.append(0)
            else:
                gt_leision.append(2)
        


    leision_states = {}
    pcm = confusion_matrix(gt_leision, hypo_leision)
    # high grade
    tp, fn, fp, tn = pcm[1, 1], pcm[1, 0]+pcm[1, 2], pcm[0, 1]+pcm[2,1], pcm[0, 0]+pcm[2,2]
    # leision_states['acc'] = (tp + tn) / (tn + fp + fn + tp)
    leision_states['precision'] = tp / (fp + tp)
    leision_states['recall'] = tp / (fn + tp)
    leision_states['specificity'] = tn / (fp + tn)
    leision_states['youden'] = leision_states['recall'] + leision_states['specificity'] - 1
    print("high grade", leision_states)

     # low grade
    tp, fn, fp, tn = pcm[0, 0], pcm[0, 1]+pcm[0, 2], pcm[1, 0]+pcm[2,0], pcm[1, 1]+pcm[2,2]
    # leision_states['acc'] = (tp + tn) / (tn + fp + fn + tp)
    leision_states['precision'] = tp / (fp + tp)
    leision_states['recall'] = tp / (fn + tp)
    leision_states['specificity'] = tn / (fp + tn)
    leision_states['youden'] = leision_states['recall'] + leision_states['specificity'] - 1
    print("low grade", leision_states)

     # all grade
    tp, fn, fp, tn = pcm[1, 1]+pcm[0, 0]+pcm[0,1]+pcm[1,0], pcm[0, 2]+pcm[1, 2], pcm[2, 0]+pcm[2,1], pcm[2,2]
    # leision_states['acc'] = (tp + tn) / (tn + fp + fn + tp)
    leision_states['precision'] = tp / (fp + tp)
    leision_states['recall'] = tp / (fn + tp)
    leision_states['specificity'] = tn / (fp + tn)
    leision_states['youden'] = leision_states['recall'] + leision_states['specificity'] - 1
    print("all cancer", leision_states)

    f1 = f1_score(gt_leision, hypo_leision, average='macro')
    acc = accuracy_score(gt_leision, hypo_leision)

    print(pcm)
    print("acc", acc)
    print("f1", f1)


# parse_tags()
calculate()
# calculate_all()