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

    def gini(self):
        return self.weight_dict().gini()

    def weight_dict(self):
        return get_class_weights(self.y)

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
    
    def gini(self):
        arr=list(self.values())
        arr.sort()
        arr=np.array(arr)
        index = np.arange(1,arr.shape[0]+1)
        n = arr.shape[0]     
        return ((np.sum((2 * index - n  - 1) * arr)) / (n * np.sum(arr))) 
#    def size_dict(self):
#        d={ i:(1.0/w_i) for i,w_i in self.items()}
#        return  WeightDict(d).norm()

def get_class_weights(y):
    params=WeightDict() 
    cats=  list(set(y))
    n_cats= len(cats) #int(max(y))+1
    params=WeightDict({cat_i:0 for cat_i in cats})
    for y_i in y:
        params[y_i]+=1
    print(params)
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
    data=read_arff("AutoML/yeast.arff")
    w=(data.weight_dict())
    print(w.gini())