import argparse
import json
import jieba
from model import *
from utils import *
from utils import eval


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default='./data/pos_train.json', type=str, required=False)
    parser.add_argument("--save_path", default='./params/', type=str, required=False)
    parser.add_argument("--train", default=False, action='store_true')
    parser.add_argument("--test", default=False, action='store_true')
    parser.add_argument("--eval", default=False, action='store_true')
    args = parser.parse_args()

    parser.add_argument("--sentence", type=str, required=True if args.test == True else False)

    if args.train:
        model = HiddenMarkovModelTrainer(args.data, args.save_path)
        model.train()

    if args.test:
        sent = jieba.lcut(args.sentence)
        init_vec_path = args.save_path + 'init-vector.json'
        transition_path = args.save_path + 'transition.json'
        emission_path = args.save_path + 'emission.json'

        with open(init_vec_path, encoding='utf-8') as f:
            init_vec = json.load(f)
        with open(transition_path, encoding='utf-8') as f:
            transition = json.load(f)
        with open(emission_path, encoding='utf-8') as f:
            emission = json.load(f)

        path = viterbi(init_vec, transition, emission, sent)

        s = ''
        i = 0
        for p in path:
            s += sent[i] + '/' + '<' + p + '>' + ' '
            i += 1
        print(s)

    if args.eval:
        evaluator = eval.Evaluator('./data/pos_test.json', args)
        precision = evaluator.get_precision()
        recall = evaluator.get_recall()
        f1 = evaluator.get_f1()
        print("---The PRF Matrics---")
        print("|\tPrecision Score\t|\tRecall Score\t|\tF-1 Score\t|")
        print("|\t{:.4f}\t\t|\t{:.4f}\t\t|\t{:.4f}\t\t|".format(precision, recall, f1))


if __name__ == '__main__':
    main()
