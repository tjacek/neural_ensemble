import numpy as np

class Dataset(object):
    def __init__(self,X,y=None):
        self.X=X
        self.y=y

    def __len__(self):
        return len(self.y)

    def dim(self):
        return self.X.shape[1]
        
    def n_cats(self):
        return int(max(self.y))+1

class WeightDict(dict):
    def __init__(self, arg=[]):
        super(WeightDict, self).__init__(arg)

    def Z(self):
        return sum(list(self.values()))

    def norm(self):
        Z=self.Z()
        for i in self:
            self[i]= self[i]/Z
        return self
    
    def size_dict(self):
        d={ i:(1.0/w_i) for i,w_i in self.items()}
        return  WeightDict(d).norm()

def get_class_weights(y):
    params=WeightDict() 
    n_cats=int(max(y))+1
    for i in range(n_cats):
        size_i=sum((y==i).astype(int))
        if(size_i>0):
            params[i]= 1.0/size_i
        else:
            params[i]=0
    return params.norm()


def read_arff(in_path:str):
    X,y=[],[]
    with open(in_path) as f:
        for line_i in f:
            if( (not '@' in line_i) and
                   (len(line_i) > 1)):
                line_i=line_i.rstrip()
                line_i=line_i.split(",")
                y.append(line_i[-1])
                X.append([float(cord_j)  for cord_j in line_i[:-1]])
    return Dataset(X=np.array(X),
                y=y)

if __name__ == '__main__':
    read_arff("AutoML/yeast.arff")