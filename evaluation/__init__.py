from evaluation.bleu import Bleu
from evaluation.meteor import Meteor
from evaluation.rouge import Rouge
from evaluation.cider import Cider
from evaluation.tokenizer import PTBTokenizer
from evaluation.spice import Spice

def compute_scores(gts, gen):
    metrics = (Bleu(), Meteor(), Rouge(), Cider(), Spice())
    all_score = {}
    all_scores = {}
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores

    return all_score, all_scores
