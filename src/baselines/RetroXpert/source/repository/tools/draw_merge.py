
import argparse

parser = argparse.ArgumentParser(description="Combine reaction drawings using the order in a reaction CSV.")
parser.add_argument("input_csv", metavar="REACTIONS.csv", help="Path to the input reaction CSV.")
file = parser.parse_args().input_csv


from os import listdir
from os.path import isfile, join

mypath='./plots'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
png_files = [f for f in onlyfiles if f.endswith('.png')]
png_files.sort()
print(len(png_files), png_files[:10])


import pandas as pd

data = pd.read_csv(file) 
rxn = data['rxn_smiles'].tolist()


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from textwrap import wrap


for i in range(1000):
    s = './plots/{}_{}.png'.format(i, 0)
    t = './plots/{}_{}.png'.format(i, 1)

    r = rxn[i]
    
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(mpimg.imread(s))
    plt.tight_layout()
    plt.axis('off')
 
    plt.subplot(1, 2, 2)
    plt.imshow(mpimg.imread(t))
    plt.tight_layout()
    plt.axis('off')
    

    plt.suptitle("\n" + "\n".join(wrap(r, 200)))
    #plt.show()
    plt.savefig('./plots_rxn/rxn_{}.png'.format(i), bbox_inches='tight')
    plt.close()

