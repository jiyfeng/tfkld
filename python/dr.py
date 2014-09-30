## dr.py
## Author: Yangfeng Ji
## Date: 09-06-2014
## Time-stamp: <yangfeng 09/11/2014 23:18:34>

from scipy.sparse import vstack, lil_matrix
from cPickle import load, dump
from scipy.sparse.linalg import svds
import numpy, gzip

class DimReduction(object):
    def __init__(self, M, K):
        """ Initialization
        """
        self.M = M
        self.K = K


    def svd(self):
        U, s, Vt = svds(self.M, k=self.K)
        W = U.dot(numpy.diag(s))
        H = Vt
        return (W, H)


    def nmf(self):
        pass


def main(K=200):
    with gzip.open("original-data.pickle.gz") as fin:
        D = load(fin)
        trnM, trnL = D['trnM'], D['trnL']
        devM, devL = D['devM'], D['devL']
        tstM, tstL = D['tstM'], D['tstL']
    # Combine data together for DR
    M = vstack([trnM, devM, tstM])
    print 'M.shape = {}'.format(M.shape)
    dr = DimReduction(M, K)
    W, H = dr.svd()
    # Split data
    trnM = W[:(2*len(trnL)), :]
    devM = W[(2*len(trnL)):(2*len(trnL+devL)), :]
    tstM = W[(2*len(trnL+devL)):, :]
    # Save data
    D = {'trnM':trnM, 'trnL':trnL, 'devM':devM, 'devL':devL,
         'tstM':tstM, 'tstL':tstL}
    print 'Save data into file ...'
    with gzip.open("dr-data.pickle.gz", "w") as fout:
        dump(D, fout)
    print 'Done'
    

if __name__ == '__main__':
    main()
