## classification.py
## Author: Yangfeng Ji
## Date: 09-06-2014
## Time-stamp: <yangfeng 09/11/2014 23:29:41>

""" Classification class
"""


from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import normalize
from cPickle import load
import numpy, gzip


class Classifier(object):
    def __init__(self, trnM, trnL, C=0.2, penalty='l2',
                 loss='l1'):
        self.clf = LinearSVC(C=C, penalty=penalty,
                             loss=loss, dual=True,
                             class_weight=None)
        self.trnM = trnM
        self.trnL = trnL

    def train(self):
        self.clf.fit(self.trnM, numpy.array(self.trnL))

    def predict(self, M, L):
        L = numpy.array(L)
        predL = self.clf.predict(M)
        acc = accuracy_score(L, predL)
        confmat = confusion_matrix(L, predL)
        f1, p, r = f1score(confmat)
        print 'Accuracy = {}, Precision = {}, Recall = {}, F1 = {}'.format(acc, p, r, f1)


def f1score(mat):
    # Column sum - precision
    p = (1.0 * mat[0,0] / mat[:,0].sum()) + (1.0 * mat[1,1] / mat[:,1].sum())
    p = p / 2.0
    # Row sum - recall
    r = (1.0 * mat[0,0] / mat[0,:].sum()) + (1.0 * mat[1,1] / mat[1,:].sum())
    r = r / 2.0
    f1 = (2 * p * r) / (p + r)
    return f1, p, r
        

def main(with_addfeat=False, with_normalize=False):
    with gzip.open("clf-data.pickle.gz") as fin:
        D = load(fin)
        trnM, trnL = D['trnM'], D['trnL']
        devM, devL = D['devM'], D['devL']
        tstM, tstL = D['tstM'], D['tstL']
    if with_addfeat:
        print 'With additional features ...'
        with gzip.open("addfeat-data.pickle.gz") as fin:
            addD = load(fin)
            trnM = numpy.hstack((trnM, addD['trnM']))
            devM = numpy.hstack((devM, addD['devM']))
            tstM = numpy.hstack((tstM, addD['tstM']))
    if with_normalize:
        trnM = normalize(trnM)
        devM = normalize(devM)
        tstM = normalize(tstM)
    print trnM.shape
    clf = Classifier(trnM, trnL)
    clf.train()
    clf.predict(devM, devL)
    clf.predict(tstM, tstL)


if __name__ == '__main__':
    main()
        
