from sklearn.neighbors import KNeighborsClassifier
import variants

class InlinerVoting(object):
    def __init__(self,k=3,min_clf=1):
        self.k=k
        self.min_clf=min_clf

    def __call__(self,inst_i,clf_type_i):
        neigh = KNeighborsClassifier(n_neighbors=self.k)
        nn_preds=[]
        for train_i,test_i in zip(inst_i.train.binary,inst_i.test.binary):
            neigh.fit(train_i, inst_i.train.targets)
            y_i= neigh.predict(test_i)#, inst_i.train.targets)
            nn_preds.append(y_i)
        