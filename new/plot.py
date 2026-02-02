import matplotlib.pyplot as plt

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

def plot_box(result_dict,data,clf_types=None):
    data.sort()
    color_map=SimpleColorMap()
    fig, ax = plt.subplots()
    if(clf_types is None):
        clf_types=result_dict.clfs()
    step=len(clf_types)
    for i,clf_i in enumerate(clf_types):
        dict_i=result_dict.get_clf(clf_i,metric="acc",split=10)
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
