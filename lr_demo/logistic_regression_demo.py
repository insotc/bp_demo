
import sys, math
from collections import defaultdict

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

def sigmoid(x):
    if x < -15.0:
        return 1e-7
    if x > 15.0:
        return 1-1e-7
    return 1/(1+math.pow(math.e, -x))

class lr_model(object):
    def __init__(self, learging_rate = 0.01, eval_interval =10):
        self.weights = defaultdict(float)
        self.learging_rate = learging_rate
        self.eval_interval = eval_interval
        self.recent_loss = []
        self.recent_label = []
        self.recent_pred = []
        self.train_iter = 0
        assert self.eval_interval > 0

    def eval_train(self, label, pred, loss):
        self.train_iter += 1
        self.recent_loss.append(loss)
        self.recent_label.append(label)
        self.recent_pred.append(1 if pred > 0.5 else 0)
        if len(self.recent_loss) == self.eval_interval:
            num_pos = sum([1 for i in range(len(self.recent_label)) if self.recent_label[i] == self.recent_pred[i]])
            print "iter:%s acc:%s loss:%s" % (self.train_iter, float(num_pos)/len(self.recent_pred) ,sum(self.recent_loss) / len(self.recent_loss))
            self.recent_loss = []
            self.recent_pred = []
            self.recent_label = []

    # sgd trainer
    # for labels, 0 for neg, 1 for pos
    def train(self, features_list, label_list):
        num_ins = len(features_list)
        assert num_ins == len(label_list)
        assert len(set(label_list) - {0,1}) == 0
        for i in range(num_ins):
            features, label = features_list[i], label_list[i]
            pred = self.predict(features)
            assert pred >= 0 and pred <= 1
            loss = -math.log(pred) if label == 1 else -math.log(1-pred)
            self.eval_train(label, pred, loss)
            grad = -1/pred if label == 1 else 1/(1-pred)
            grad *= pred*(1-pred)
            for fea in features:
                fea_grad = grad * features[fea]
                self.weights[fea] -= self.learging_rate*fea_grad


    def predict(self, features):
        weight_sum = 0.0
        for fea in features:
            if fea in self.weights:
                weight_sum += self.weights[fea]*features[fea]
        return sigmoid(weight_sum)

    def eval(self, features_list, labels):
        pred = [self.predict(features) for features in features_list]
        pred = map(lambda x: 1 if x>0.5 else 0, pred)
        print "global acc: %s" % precision_score(labels, pred, average='micro')


def test_train():
    breast_cancer = datasets.load_breast_cancer()
    features = breast_cancer.data
    labels = breast_cancer.target
    features_list = []
    for i in range(len(features)):
        features_list.append({fea_idx: features[i][fea_idx] for fea_idx in range(len(features[i]))})

    model = lr_model()
    model.train(features_list, labels)

def test_train_and_eval():
    breast_cancer = datasets.load_breast_cancer()
    features = breast_cancer.data
    labels = breast_cancer.target
    features_list = []
    for i in range(len(features)):
        features_list.append({fea_idx: features[i][fea_idx] for fea_idx in range(len(features[i]))})

    x_train, x_test, y_train, y_test = train_test_split(features_list, labels, test_size=0.2)

    model = lr_model(learging_rate=0.05, eval_interval=100)
    print "before train:"
    model.eval(x_test, y_test)
    model.train(x_train, y_train)
    print "after train:"
    model.eval(x_test, y_test)

if __name__ == "__main__":
    # test_train()
    test_train_and_eval()


