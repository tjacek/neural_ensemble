import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import variants

class InlinerVoting(object):
    def __init__(self,k=5,min_clf=2):
        self.k=k
        self.min_clf=min_clf

    def __call__(self,inst_i,clf_type_i):
        neigh = KNeighborsClassifier(n_neighbors=self.k)
        nn_preds=[]
        for train_i,test_i in zip(inst_i.train.binary,inst_i.test.binary):
            neigh.fit(train_i, inst_i.train.targets)
            y_i= neigh.predict(test_i)#, inst_i.train.targets)
            nn_preds.append(y_i)
        nn_preds=np.array(nn_preds).T
        clfs=variants.train_clfs(clf_type_i,inst_i.train)
        votes=variants.eval_clfs(clfs,inst_i.test)
        common_pred=variants.common_variant(inst_i,clf_type_i)
        print(common_pred.shape)
        inliner_pred=[]
        for i,nn_i in enumerate(nn_preds):
            cand_i=[ vote_j[i] for vote_j in votes]
            pred_i=[ np.argmax(vote_j) for vote_j in cand_i]
            s_votes=[ vote_j for j,vote_j in enumerate(cand_i)
                       if(nn_i[j]==pred_i[j]) ]
            if(len(s_votes)>self.min_clf):
#                k=np.argmax(np.sum(s_votes,axis=1),axis=0)
                sum_i=np.sum(s_votes,axis=0)
                cat_i= np.argmax(sum_i,axis=0)
                inliner_pred.append(cat_i)
            else:
            	inliner_pred.append( common_pred[i])
#        print(list(zip(inliner_pred,inst_i.get_true())))
        return np.array(inliner_pred)
#            print(nn_i)
#            print(pred_i)