from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import os

def smoothing(x,smooth=0.99):
    x = [(i.step,i.value) for i in x]
    x[0]=x[0][1]
    for i in range(1,len(x)):
        x[i] = (x[i-1] * smooth + x[i][1] * (1 - smooth))
    return x

def plot(x:dict, smooth=0.99, legend_pos='lower right', figsize=(10, 6), ymin=None, ymax=None):
    plt.clf()
    fig, ax1 = plt.subplots(1, 1, figsize=figsize) 
    ax1.spines['top'].set_visible(False)                   
    ax1.spines['right'].set_visible(False)  

    num_colors = len(x)
    colors = plt.get_cmap('tab20c')  

    sorted_items = sorted(x.items())  

    for i, (k, v) in enumerate(sorted_items):
        color = colors(i % num_colors)  
        ax1.plot(smoothing(v, smooth=smooth), label=k, color=color)

    plt.legend(loc=legend_pos)
    
    if ymin is not None:
        ax1.set_ylim(ymin=ymin)  
    if ymax is not None:
        ax1.set_ylim(ymax=ymax)  
    return ax1

def getData(path,v=None):
    res={}
    ea=event_accumulator.EventAccumulator(path) 
    ea.Reload()
    try:
        if not v is None:
            return ea.scalars.Items(v)
        for k in ea.scalars.Keys():
            res[k]=ea.scalars.Items(k)
        return res
    except:
        return None

def draw(src,dest,prefix,smooth=0.6,figsize=(10,6),ymin=None,ymax=None):
    data=getData(src)
    ax=plot(data,smooth=smooth,legend_pos='upper right',figsize=figsize,ymax=ymax,ymin=ymin)
    ax.set_xlabel("epoch")
    ax.set_title(f"{prefix}")
    plt.savefig(fname=f'{dest}/{prefix}.png', format='png',dpi=300)
    plt.close()

def joindraw(src:dict,dest,prefix,smooth=0.6,figsize=(10,6),ymin=None,ymax=None):
    data={}
    for k,v in src.items():
        res = getData(v,prefix.split('_')[0])
        if res:data[k]=res
    ax=plot(data,smooth=smooth,legend_pos='best',figsize=figsize,ymax=ymax,ymin=ymin)
    plt.legend(ncol=3)
    ax.set_xlabel("epoch")
    ax.set_title(f"{prefix}")
    plt.savefig(fname=f'{dest}/{prefix}.png', format='png',dpi=300)
    plt.close()

def findFile(src, type='Acc'):
    res = {}  
    for entry in os.listdir(src):  
        if os.path.isdir(os.path.join(src, entry)): 
            exp_path = os.path.join(src, entry) 
            log_path = os.path.join(exp_path, 'log')
            for sub_entry in os.listdir(log_path): 
                if os.path.isdir(path:=os.path.join(log_path, sub_entry)) and sub_entry.startswith(type):
                    res[entry]=os.path.join(path,os.listdir(path)[0])
                elif sub_entry.startswith(type):
                    res[entry]=path
    return res

if __name__ == '__main__':
    def process_and_draw(file_path, prefix, smooth=0.99, output_dir='ana', ymin=None, ymax=None):
        src = findFile('exp', file_path)
        joindraw(src, output_dir, f'{prefix}', smooth)
        if ymin:
            for y_min in ymin:
                joindraw(src, output_dir, f'{prefix}_{y_min}', smooth, ymin=y_min)
        if ymax:
            for y_max in ymax:
                joindraw(src, output_dir, f'{prefix}_{y_max}', smooth, ymax=y_max)
    plt.savefig(fname='ana/111.png')
    print(1)
    process_and_draw('events', 'Loss', 0.99, ymax=[0.1,0.2])
    process_and_draw('Acc_train', 'Acc_train', 0.99, ymin=[0.93, 0.96, 0.97])
    process_and_draw('Acc_val', 'Acc_val', 0.99, ymin=[0.93, 0.96, 0.97])
    process_and_draw('Acc_filtered_val_last', 'Acc_filtered_val_last', 0.99, ymin=[0.93, 0.96, 0.97])
    process_and_draw('Iou_train', 'Iou_train', 0.99, ymin=[0.6, 0.7])
    process_and_draw('Iou_val', 'Iou_val', 0.99, ymin=[0.6, 0.7])
