import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import learn,binary

class OneVsOne(binary.NeuralEnsemble):
    def fit(self,X,y):
        pairs=self.select_pairs(X,y)
        self.extractors,self.models=[],[]
        for i,j in pairs:
            selected=[k for k,y_k in enumerate(y)
                        if((y_k==i) or (y_k==j))]
            y_s=[ int(y[k]==i) for k in selected]                
            y_s= tf.keras.utils.to_categorical(y_s, num_classes = 2)
            X_s=np.array([ X[k,:] for k in selected])
            extractor_i=self.train_extractor(X_s,y_s)
            binary_i=extractor_i.predict(X)
            self.train_model(X,binary_i,y)

    def select_pairs(self,X,y):
        train,test=split_dataset(X,np.array(y))
        cf=self.make_cf(train,test)
        n_cats= max(y)+1
        cat_size={ cat_i: y.count(cat_i) 
            for cat_i in range(n_cats)}
        pairs,values=[],[]
        for i in range(n_cats):
            for j in range(n_cats):
                if(i!=j):
                    size_ij=min(cat_size[i],cat_size[j])
                    cf_ij=cf[i][j]/float(size_ij)
                    values.append(cf_ij)
                    pairs.append((i,j))
        pairs=[ pairs[i]
            for i in np.argsort(values)[-n_cats:]]
        return pairs

    def make_cf(self,train,test):
        clf=learn.get_clf(self.multi_clf)
        clf.fit(train[0],train[1])
        y_pred=clf.predict(test[0])
        return confusion_matrix(y_pred,test[1])

def split_dataset(X,y):
    skf = StratifiedKFold(n_splits=2)
    splits=[ split_i
        for split_i in skf.split(X, y)]
    train,test=splits
    train_X,train_y=X[train[0]],y[train[0]]
    test_X,test_y=X[test[0]],y[test[0]]
    return (train_X,train_y),(test_X,test_y)