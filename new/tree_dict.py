import numpy as np
from scipy.stats import entropy 


class TreeDict(dict):
    def __init__(self, arg=[]):
        super(TreeDict, self).__init__(arg)
        self.targe_dist=None
        self.n_samples=None
    
    def get_attr(self,attr,s_nodes):
        return [ self[i](attr) for i in s_nodes]
    
    def mutual_info(self):
        h_y=entropy(self.target_dist)
        by_cat=self.target_dist*self.n_samples
        info,nodes=[],[]
        for i,node_i in self.items():
            p_target_1=node_i.right_dist
            p_target_0=node_i.get_dist(by_cat)
            p_1= node_i.get_p(self.n_samples)
            p_0=1.0-p_1
            h_node_y  =  log_helper(p_target_1, p_1)
            h_node_y +=  log_helper(p_target_0, p_0)
            i_xy= h_y - h_node_y
            info.append(i_xy)
            nodes.append(i)
        return info,nodes

    def get_paths(self,s_nodes):
        indexes=[]
        for path_j in self.indv_paths(s_nodes):     
            indexes+=path_j 
        return list(set(indexes))

    def indv_paths(self,s_nodes):
        for i in s_nodes:
            node_i=self[i]
            path_j=[i]
            current=node_i
            while(current.parent!= (-2)):
                path_j.append(current.parent)
                current=self[current.parent]
            yield path_j

class NodeDesc(object):
    def __init__( self,
                  parent,
                  feature,
                  threshold,
                  right_dist,
                  n_samples):
        self.parent=parent
        self.feature=feature
        self.threshold=threshold
        self.right_dist=right_dist
        self.n_samples=n_samples
    
    def get_dist(self,by_cat):
        dist_i=self.right_dist*self.n_samples
        diff_i=by_cat-dist_i
        diff_i/=np.sum(diff_i)
        return diff_i

    def get_p(self,total_samples):
        return self.n_samples/total_samples

    def __call__(self,attr):
        if(attr=="threshold"):
            return self.threshold
        if(attr=="feat"):
            return self.feature

def make_tree_dict(clf):
    tree_dict=TreeDict()
    raw_tree=clf.tree_
    parent=find_params(raw_tree)    
    root=np.argmin(parent)
    tree_dict.target_dist= raw_tree.value[root][0] 
    tree_dict.n_samples=raw_tree.weighted_n_node_samples[root]
    for i,parent in enumerate(parent):
        if(parent!=(-1)):
            right_i=raw_tree.children_right[i]
            right_dist=raw_tree.value[right_i][0]
            n_samples=raw_tree.weighted_n_node_samples[right_i]
            node_i=NodeDesc(parent=parent,
                            feature=raw_tree.feature[i],
                            threshold=raw_tree.threshold[i],
                            right_dist=right_dist,
                            n_samples=n_samples)
            tree_dict[i]=node_i            
    return tree_dict

def find_params(raw_tree):
    n_nodes=len(raw_tree.feature)
    parent= -2*np.ones((n_nodes,),dtype=int)
    for i in range(n_nodes):
        left_i=raw_tree.children_left[i]
        right_i=raw_tree.children_right[i]
        if(left_i<0 or right_i <0):
            parent[i]= -1
            continue
        parent[left_i]=  i
        parent[right_i]= i
    return parent

def log_helper(p_target_node, p_node):
    if(p_node==0):
        return 0
    total=0
    for i,p_i in enumerate(p_target_node):
        if(p_i==0):
            continue
        quot_i=p_i/p_node
        if(quot_i<=0):
            continue
        total+= p_i*np.log(quot_i)
    return -total
