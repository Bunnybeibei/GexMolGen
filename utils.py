import random

import torch
import numpy as np

from hgraph import common_atom_vocab
from hgraph.mol_graph import MolGraph

import rdkit
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.Fraggle import FraggleSim


def to_numpy(tensors):
    convert = lambda x : x.numpy() if type(x) is torch.Tensor else x
    a,b,c = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c


def tensorize(mol_batch, vocab):
    x = MolGraph.tensorize(mol_batch, vocab, common_atom_vocab)
    return to_numpy(x)


def tensorize_change(mol_batch, vocab):
    x = MolGraph.tensorize(mol_batch, vocab, common_atom_vocab)
    return to_numpy(x)


def set_seed(seed):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class molCalSim(object):
    def __init__(self, ref_smile, gen_smiles):
        """
        Calculate molecular similarity.
        :param ref_smile: [bs,]
        :param gen_smiles: [candidate_nums, bs]
        """
        self.ref_smile = ref_smile
        self.ref_num = len(ref_smile)
        self.bw = len(gen_smiles)
        print('There are {0} kinds of drug, and the beam width is {1}.'.format(self.ref_num, self.bw))

    def Morgan_Tanimoto(self, ref_smile, gen_smiles):
        # MorganFingerprint
        ref_mol = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(ref_smile), 3)
        gen_mol = [Chem.MolFromSmiles(smiles) for smiles in gen_smiles]
        gen_fp = [AllChem.GetMorganFingerprint(mol, 3) for mol in gen_mol]

        # Similarity
        similarity_all = []
        for i in range(len(gen_fp)):
            similarity = DataStructs.TanimotoSimilarity(ref_mol, gen_fp[i])
            similarity_all.append(str(round(similarity, 2)))

        return gen_smiles[np.argmax(similarity_all)], max(similarity_all)

    def Fraggle(self, ref_smile, gen_smiles):
        # FraggleFingerprint
        ref_fp = Chem.MolFromSmiles(ref_smile)
        gen_fp = [Chem.MolFromSmiles(smiles) for smiles in gen_smiles]

        # Similarity
        similarity_all = []
        for fp in gen_fp:
            try:
                similarity = FraggleSim.GetFraggleSimilarity(ref_fp, fp)[0]
            except:
                similarity_all.append(str(round(0., 2)))
                continue
            similarity_all.append(str(round(similarity, 2)))

        return gen_smiles[np.argmax(similarity_all)], max(similarity_all)

    def MACCS_Tanimoto(self, ref_smile, gen_smiles):
        ref_mol = Chem.MolFromSmiles(ref_smile)
        ref_fp = MACCSkeys.GenMACCSKeys(ref_mol)

        gen_mols = [Chem.MolFromSmiles(smiles) for smiles in gen_smiles]

        similarity_all = []
        for gen_mol in gen_mols:
            gen_fp = MACCSkeys.GenMACCSKeys(gen_mol)
            similarity = DataStructs.TanimotoSimilarity(ref_fp, gen_fp)
            similarity_all.append(str(round(similarity, 2)))

        return gen_smiles[np.argmax(similarity_all)], max(similarity_all)

    def main(self, query_smiles, method='Morgan'):
        result_smiles = []
        result_scores = []
        for i in range(self.ref_num):
            if method == 'Morgan':
                morgan_smiles, morgan_metric = self.Morgan_Tanimoto(self.ref_smile[i], \
                                                                    [bs_smiles[i] for bs_smiles in query_smiles])
            elif method == 'MACCS':
                morgan_smiles, morgan_metric = self.MACCS_Tanimoto(self.ref_smile[i], \
                                                                    [bs_smiles[i] for bs_smiles in query_smiles])
            elif method == 'Fraggle':

                morgan_smiles, morgan_metric = self.Fraggle(self.ref_smile[i], [bs_smiles[i] for bs_smiles in query_smiles])
            result_smiles.append(morgan_smiles)
            result_scores.append(morgan_metric)

        return result_smiles, result_scores


if __name__ == "__main__":
    """
    The "ref_smiles" and "gen_smiles" mentioned below are derived from the OCR molecule recognition in the WGAN paper.
    """
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    ref_smile = ['O=C1C(=O)C(Nc2ccncc2)C1NCc1ccccc1F',\
                 'CN1CCCC1COc1cncc(CCc2ccccc2)c1',\
                 'N=C1C=CC(NC(=O)c2ccc(Cl)cc2)=C/C1=C/N',\
                 'CC[C@H](C)[C@@H](/C=C/c1ccccc1)NC(=O)[C@@H]1CCCCN1CC(=O)c1ccccc1',\
                 'O=C(NCCCc1ccccc1)c1cc(NCc2cc(O)ccc2O)ccc1O',\
                 'CCSC(=S)SCC(=O)c1ccc(C(=O)NCc2ccccc2)cc1',\
                 'OC1=NC2NC=C(c3ccc(O)cc3)N=C2N1Cc1ccccc1',\
                 'CC1CN(c2cc(O)nc(NCc3cccc4ccccc34)n2)CCO1',\
                 'CCCNS(=O)C1=CC=[SH]C(Br)=C1',\
                 'COc1ccc(OCCOC(=O)Nc2ccccc2)cc1'
                 ]
    gen_smiles = [['CC(=O)N1C[C@H](CN(C)C)[C@H]1NCc1ccccc1F',\
                 'COc1ccc(O[C@]2(CCl)CCCN2C[C@@H]2CCCN2C)n(CC2CN(Cc3ccccc3)C2)c1=O',\
                 'Cc1ccc(C)c(C(=O)Nc2ccc3nc(C)ccc3c2)c1',\
                 'C=Nc1cnc(NC(=O)[C@@H]2CCCN(CC(=O)c3ccccc3)C(OC(C)=O)CC2)cc1O',\
                 'COCN/C(=C\C(=N)C(=O)NCCCc1ccccc1)NCC1OC(F)(F)OCC1C',\
                 'CCN1C[C@H](C)N1C(=O)NCc1ccccc1',\
                 'CC1=Cc2nc(O)n(Cc3ccccc3)c21',\
                 'CC(=O)N1CC[C@H]1Nc1cccc(F)c1.NC(=O)C1CCN(Cc2cccc3ccccc23)C1C(=O)O',\
                 'CC1(C)CC1C(O)c1cccc(Br)c1',\
                 'C=C(COC(=O)Nc1ccccc1)OC(=O)CC'
                 ]]

    best_smile, max_similarity = molCalSim(ref_smile, gen_smiles).main(gen_smiles,method='MACCS')
    print(best_smile)
    print(max_similarity)