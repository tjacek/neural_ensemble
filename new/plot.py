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