import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
import pred,utils

class SimpleColorMap(object):
    def __init__(self,colors=None):
        if(colors is None):
            colors=['lime','red','blue','tomato',
                    'orange','skyblue','peachpuff',
                    'yellow','black' ]
        self.colors=colors

    def __call__(self,i):
        return self.colors[i % len(self.colors)]
    
    def get_handlers(self):
        return [plt.Rectangle((0,0),1,1, color=color_i) 
                    for color_i in self.colors]

def simple_plot(x,y,title="",xlabel="x"):
    fig, ax = plt.subplots()
    ax.plot(x,y)
    ax.set_title(title,fontsize=10)
    plt.xlabel(xlabel)
    plt.show()

def multi_plot(plot_dict,
               xlabel="Accuracy",
               ylabel="Density"):
    fig, ax = plt.subplots()
    color_map=SimpleColorMap()
    for i,id_i in enumerate(plot_dict.keys()):
        x_i,y_i=plot_dict[id_i]
        ax.plot(x_i,y_i,color=color_map(i),label=id_i)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def group_plot(plot_dict,
               xlabel="Accuracy",
               ylabel="Density"):
    fig, ax = plt.subplots()
    color_map=SimpleColorMap()
    for i,(type_i,labels_i)  in enumerate(plot_dict.items()):
#            output_i=helper(in_path,labels_i)
            for x_j,y_j in labels_i.values(): 
                ax.plot(x_j,y_j,
                       color=color_map(i),
                       label=type_i)#,
    plt.legend()
    plt.show()    

def compute_density(value,x=None,n_steps=100):
    value=value.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(value)
    if(x is None):
        a_max,a_min=np.max(value),np.min(value)
        delta= (a_max-a_min)/n_steps
        x=np.arange(n_steps)*delta
        x+=a_min
    log_dens= kde.score_samples(x.reshape(-1, 1))
    dens=np.exp(log_dens)
    return x,dens

def plot_box(result_dict,data,clf_types=None):
    data.sort()
    color_map=SimpleColorMap()
    fig, ax = plt.subplots()
    if(clf_types is None):
        clf_types=result_dict.clfs()
#    clf_types=["RF","GRAD","MLP","TREE-MLP","TREE-ENS"]
    step=len(clf_types)
    for i,clf_i in enumerate(clf_types):
        dict_i=result_dict.get_clf(clf_i,metric="acc")
        values_i=[dict_i[data_j] for data_j in data]
        positions_i=[j*step+i for j,_ in enumerate(data)]

        box_i=ax.boxplot(values_i,
                         positions=positions_i,
                         patch_artist=True)
        plt.setp(box_i['medians'], color="black")
        plt.setp(box_i['boxes'], color=color_map(i))
    legend_handles = color_map.get_handlers()
    ax.legend(legend_handles,clf_types)
    plt.ylabel("Accuracy")
    offset=int(step/2)
    xticks=[ (i*step) 
             for i,_ in enumerate(data)]
    plt.xticks(xticks, minor=True)
    xticks=[offset+i for i in xticks]
    plt.xticks(xticks, data,rotation='vertical')
    plt.grid(which='minor')
    plt.tight_layout()
    plt.show()

def make_plots(conf_path):
    conf=utils.read_json(conf_path)
    for exp_i,dict_i in conf.items():
        result_i=eval.get_result_dict(dict_i["exp_path"])
        for data_j in dict_i["data"]:
            plot_box(result_i,data_j,dict_i["clf_types"])

def dict_plot(x_dict,
              y_dict,
              xlabel,
              ylabel,
              text=True,
              title=None):
    fig=plt.figure()
    if(type(text)==dict):
        for data_i in x_dict:
            plt.text(x_dict[data_i], 
                     y_dict[data_i], 
                     text[data_i],
                     fontdict={'weight': 'bold', 'size': 9})
    elif(text):
        for data_i in x_dict:
            plt.text(x_dict[data_i], 
                     y_dict[data_i], 
                     data_i,
                     fontdict={'weight': 'bold', 'size': 9})
    else:
        data=x_dict.keys()
        x=[x_dict[data_i] for data_i in data]
        y=[y_dict[data_i] for data_i in data] 
        plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(0.9*min(x_dict.values()),
             1.1*max(x_dict.values()))
    plt.ylim(0.9*min(y_dict.values()),
             1.1*max(y_dict.values()))
    plt.axline((0, 0), (1, 1))
    if(title):
        plt.title(title)
    plt.grid()
    plt.show()

if __name__ == '__main__':
    make_plots("plot.json")