
import argparse

parser = argparse.ArgumentParser(description="Extract functional groups from a reaction CSV.")
parser.add_argument("input_csv", metavar="REACTIONS.csv", help="Path to the input reaction CSV.")
file = parser.parse_args().input_csv

import pandas as pd


data = pd.read_csv(file) 

prod_smiles = data['prod_smiles'].tolist()
split =  data['dataset'].tolist()
rxn = data['rxn_smiles'].tolist()
products = {'train': [], 'test': [], 'val': []}

for s, product in zip(split, prod_smiles):
    products[s].append(product)

print(len(products['train']), len(products['test']), len(products['val']))
    

import os
import sys
# %matplotlib inline
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'IFG'))
import ifg


for idx, smiles in enumerate(products['train']):
    mol = Chem.MolFromSmiles(smiles)
    [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    fgs = ifg.identify_functional_groups(mol)
    
    highlights = []
    for fg in fgs:
        highlights += fg.atomIds

    fig = Draw.MolToMPL(mol, size=(1000, 1000), highlightAtoms=highlights)
    fig.savefig('plots_fg/{}.png'.format(idx), bbox_inches='tight')
    fig.clf()

