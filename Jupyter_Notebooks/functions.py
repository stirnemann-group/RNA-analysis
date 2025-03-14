def plot_parallel_hists(Counts_Array, Bins_Array, Labels, Colors, xlabel, title, filename, Lines=None, xlim=None):
    import numpy as np
    import matplotlib.pyplot as plt
    if not len(Colors)==len(Labels)==np.shape(Counts_Array)[0]:
        print(len(Colors),len(Labels),np.shape(Counts_Array)[0])
        print("Dimensions do not correspond.\n Size of Colors and Labels must correspond to the number of histograms to be plotted, i.e. the number of lines in Counts_Array.")
        return None
    if not np.shape(Counts_Array)[1]==np.shape(Counts_Array)[1]:
        print(np.shape(Counts_Array)[1],np.shape(Counts_Array)[1])
        print("Dimensions do not correspond.\n Abscissa (nb of nc in Bins_Array) and ordonnates (nb of nc in Counts_Array) do not correspond).")
        return None
    
    plt.figure(figsize=(20,10))
    ax = plt.subplot(1,1,1)
    plt.xlabel(xlabel,fontsize=15) #A M
    plt.yticks([],fontsize=20)
    plt.xticks(fontsize=30)
    plt.title(title,fontsize=25,pad=30)
    plt.yticks([])
    
    J=[0.65,0.65,0.65,0.65,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7]
    jump=0
    nl,nc=np.shape(Counts_Array)#nb de hists, nb de bins par hists
    gap=np.max(Counts_Array) +1/nl*np.max(Counts_Array)
    
    print("len J = ",len(J))
    print("len labels = ", len(Labels))
    
    for l in range(0,nl):
        hist=np.zeros((nc,2))#On crée un hist virtuel ici, devrait être notre histogram
        hist[:,0]=Bins_Array[l][1:]
        hist[:,1]=Counts_Array[l]+jump
        ax.plot([np.min(Bins_Array),np.max(Bins_Array)],[jump,jump],color='black')
        ax.plot(hist[:,0],hist[:,1], linewidth = 2, linestyle = 'solid',color=Colors[l])
        ax.fill_between(hist[:,0],jump,hist[:,1], alpha=0.3,color=Colors[l])
        plt.text(np.min(Bins_Array)+0.1, jump+J[l]*gap, Labels[l], fontsize = 13) #TO MODIFY IF NECESSARY
        jump-=gap
    
    if Lines!=None:
        Linescolors=['r','g','orange','b','purple','c']
        for li in range(len(Lines)):
            plt.axvline(Lines[li][1],label=Lines[li][0],linestyle='--',color=Linescolors[li])
    if xlim != None:
        plt.xlim(xlim)
    plt.legend(bbox_to_anchor=(0.01, l/(2*nl)),fontsize=20)   
    plt.grid(linestyle='--',linewidth=0.8)
    plt.savefig(filename) 
    plt.show()
