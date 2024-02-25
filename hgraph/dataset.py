import torch
from torch.utils.data import Dataset
from rdkit import Chem
import os, random, gc
import pickle
import pandas as pd
import numpy as np

from hgraph.chemutils import get_leaves
from hgraph.mol_graph import MolGraph


class MoleculeDataset(Dataset):

    def __init__(self, data, vocab, avocab, batch_size):
        safe_data = []
        for mol_s in data:
            hmol = MolGraph(mol_s)
            ok = True
            for node,attr in hmol.mol_tree.nodes(data=True):
                smiles = attr['smiles']
                ok &= attr['label'] in vocab.vmap
                for i,s in attr['inter_label']:
                    ok &= (smiles, s) in vocab.vmap
            if ok: 
                safe_data.append(mol_s)

        print(f'After pruning {len(data)} -> {len(safe_data)}') 
        self.batches = [safe_data[i : i + batch_size] for i in range(0, len(safe_data), batch_size)]
        self.vocab = vocab
        self.avocab = avocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return MolGraph.tensorize(self.batches[idx], self.vocab, self.avocab)


class MolEnumRootDataset(Dataset):

    def __init__(self, data, vocab, avocab):
        self.batches = data
        self.vocab = vocab
        self.avocab = avocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        mol = Chem.MolFromSmiles(self.batches[idx])
        leaves = get_leaves(mol)
        smiles_list = set( [Chem.MolToSmiles(mol, rootedAtAtom=i, isomericSmiles=False) for i in leaves] )
        smiles_list = sorted(list(smiles_list)) #To ensure reproducibility

        safe_list = []
        for s in smiles_list:
            hmol = MolGraph(s)
            ok = True
            for node,attr in hmol.mol_tree.nodes(data=True):
                if attr['label'] not in self.vocab.vmap:
                    ok = False
            if ok: safe_list.append(s)
        
        if len(safe_list) > 0:
            return MolGraph.tensorize(safe_list, self.vocab, self.avocab)
        else:
            return None


class MolPairDataset(Dataset):

    def __init__(self, data, vocab, avocab, batch_size):
        self.batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
        self.vocab = vocab
        self.avocab = avocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        x, y = zip(*self.batches[idx])
        x = MolGraph.tensorize(x, self.vocab, self.avocab)[:-1] #no need of order for x
        y = MolGraph.tensorize(y, self.vocab, self.avocab)
        return x + y


class DataFolder(object):

    def __init__(self, data_folder, batch_size, shuffle=True, length=100):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.length = length

    def __len__(self):
        return len(self.data_files) * self.length

    def __iter__(self):
        for fn in self.data_files:
            print('Now turn to {}'.format(fn))
            fn = os.path.join(self.data_folder, fn)
            with open(fn, 'rb') as f:
                batches = pickle.load(f)

            if self.shuffle: random.shuffle(batches) #shuffle data before batch
            for batch in batches:
                yield batch

            del batches
            gc.collect()

class DataFolder_new(object):

    def __init__(self, config):
        with open(str(config['data_drug_dir']), 'rb') as f:
            self.data_drug = pickle.load(f)

        self.data_genes = pd.read_csv(str(config['data_gene_dir']), index_col=0)
        _, self.data_genes['cell_id'] = np.unique(np.array(self.data_genes['cell_id']), return_inverse=True)
        _, self.data_genes['pert_idose'] = np.unique(np.array(self.data_genes['pert_idose']), return_inverse=True)

        self.batch_size = int(config['batch_size'])
        self.iter = -1

    def __len__(self):
        return len(self.data_drug)

    def __iter__(self):
        rand_starts = [random.randint(0, len(self.data_drug)-2) for _ in range(len(self.data_drug)-1)]
        for rand_start in rand_starts:
            flag = True
            for index in range(rand_start*self.batch_size, min((rand_start+1)*self.batch_size, self.data_genes.shape[0])):
                full_seq = self.data_genes.iloc[index, 2:].values
                full_seq = torch.from_numpy(full_seq)
                if flag:
                    seq = torch.Tensor(sorted(zip(full_seq, torch.arange(len(full_seq))))).unsqueeze(dim=0)
                    flag = False
                else:
                    seq = torch.cat((seq, torch.Tensor(sorted(zip(full_seq, torch.arange(len(full_seq))))).unsqueeze(dim=0)),dim=0)
            yield self.data_drug[rand_start], seq