import pandas as pd
import numpy as np

import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import os
import pickle
from PIL import Image
import torch.nn.functional as F
from data import ModelLoader
from data import dataPrepare
from data.model_loader import Genetic_Encoder

from utils import molCalSim, set_seed, tensorize
from hgraph import PairVocab

import torch
import argparse

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0)
parser.add_argument('--amp', default=True)
parser.add_argument('--mask_value', default=-1)
parser.add_argument('--batch_size', default=100)
parser.add_argument('--faci_n_layers', default=4)

parser.add_argument('--dropout', default=0.2)
parser.add_argument('--device', default=torch.device('cuda'))
parser.add_argument('--geneModel_file', default=r'weight/scGPT_origin/')
parser.add_argument('--genetic_vocab_file', default=r'weight/scGPT_origin/vocab.json')
parser.add_argument('--vocab_drug', default=r'weight/molecular/scaffold.txt')
parser.add_argument('--buckets', default='weight/molecular/buckets_1024.npy')
parser.add_argument('--pca_mean', default='weight/molecular/pca_mean.npy')
parser.add_argument('--pca_comp', default='weight/molecular/pca_component.npy')
parser.add_argument('--drugModel_file', default=r'weight/molecular/decoder/drug.pt')  # Details see 1..py
parser.add_argument('--geneModel_path', default=r'weight/genetic/alignment/alignment.pt')  # Details see 2..py
parser.add_argument('--facilitator_path', default=r'weight/genetic/facilitator/facilitator.pt')  # Details see 3..py

parser.add_argument('--pad_token', default="<pad>")
parser.add_argument('--granu', default=3)

args = parser.parse_args()

# fix seed
set_seed(args.seed)

# Load Model

# # Load DrugModel
model_loader = ModelLoader(drugModel_file=args.drugModel_file, geneModel_file=args.geneModel_file)
drugModel = model_loader.DrugLoader(drug_vocab_file=args.vocab_drug)
drugModel = drugModel.to(args.device)

# # Load GeneModel
model = model_loader.GeneLoader(args=args)
facilitator = model_loader.FaciLoader(args=args, path=args.facilitator_path)
model.drug_decoder = facilitator
model = model.to(args.device)

buckets = torch.from_numpy(np.load(args.buckets)).to(args.device)
pca_mean = torch.from_numpy(np.load(args.pca_mean)).to(args.device)
pca_comp = torch.from_numpy(np.load(args.pca_comp)).to(args.device)


def discrete2continue(discrete_emb, buckets, pca_mean, pca_comp):
    """Inverse Transformation of PCA"""
    continuous_emb = torch.zeros_like(discrete_emb, dtype=torch.float64)
    for i in range(continuous_emb.shape[0]):
        for j in range(continuous_emb.shape[1]):
            continuous_emb[i][j] = buckets[int(discrete_emb[i][j])]
    continuous_emb = torch.matmul(continuous_emb, pca_comp) + pca_mean.unsqueeze(0)
    return continuous_emb


def generator(control,
              pert,
              beam_width=2,
              seed=0):
    """

    Args:
        control: Gene expression data from control groups
        pert: Gene expression data from pert groups
        gene_name: (optional) Gene categories
        beam_width: The number of candidate molecules to generate
        seed: Different seeds yield different results

    Returns:
        hit_like_smiles: A List containing candidate molecules
        hit_like_mols: Molecules in RDKit's Mol format
    """

    # Fix seed
    set_seed(seed)

    # Load Dataset
    control = pd.read_csv(control.name, index_col=0, nrows=None) # sample_num * gene_num
    pert = pd.read_csv(pert.name, index_col=0, nrows=None) # sample_num * gene_num

    # # The gene categories in the control group and the experimental group need to be the same.
    if control.columns.tolist() != pert.columns.tolist():
        raise ValueError("The column names of the two CSV files do not match.")

    data_prepare= dataPrepare(genetic_vocab_file=args.genetic_vocab_file, batch_size=args.batch_size)
    data_loader = data_prepare.main(control=control, pert=pert)

    # Inference
    drugModel.eval()
    model.eval()

    hit_like_smiles = [[] for _ in range(beam_width)]
    hit_like_mols = [[] for _ in range(beam_width)]
    with torch.no_grad():
        for batch, batch_data in enumerate(data_loader):

            input_gene_ids = batch_data["genes"].cuda()
            input_values = batch_data["values"].cuda().float()
            src_key_padding_mask = torch.zeros_like(input_gene_ids, dtype=torch.bool).cuda()

            seq_length = input_values.shape[-1]
            with torch.cuda.amp.autocast(enabled=True):
                output_ctl, \
                output10_ctl, \
                output100_ctl, \
                output1000_ctl = Genetic_Encoder(model=model,
                                                 input_values=input_values[:, :seq_length // 2],
                                                 input_gene_ids=input_gene_ids[:, :seq_length // 2],
                                                 src_key_padding_mask=src_key_padding_mask[:, :seq_length // 2])

                output_desir, \
                output10_desir, \
                output100_desir, \
                output1000_desir = Genetic_Encoder(model=model,
                                                   input_values=input_values[:, seq_length // 2:],
                                                   input_gene_ids=input_gene_ids[:, seq_length // 2:],
                                                   src_key_padding_mask=src_key_padding_mask[:, seq_length // 2:])

                cell_feat = output_desir - output_ctl
                prediction = model.drug_decoder.predict(cell_feat, model.drug_embed, beam_width=beam_width)

            continuous_emb = discrete2continue(torch.from_numpy(np.array(prediction)).cuda(), buckets, pca_mean, pca_comp)
            continuous_emb = continuous_emb.float().reshape((beam_width, -1, 64))

            for i in range(beam_width):
                generate_candidates = drugModel.sample(continuous_emb[i], greedy=True)
                hit_like_mols[i].extend(generate_candidates[0])
                hit_like_smiles[i].extend(generate_candidates[1])

            torch.cuda.empty_cache()

    return hit_like_smiles, hit_like_mols


def generator_local(control,
              pert,
              beam_width=2,
              seed=0):
    """

    Args:
        control: Gene expression data from control groups
        pert: Gene expression data from pert groups
        beam_width: The number of candidate molecules to generate
        seed: Different seeds yield different results

    Returns:
        hit_like_smiles: A List containing candidate molecules
        hit_like_mols: Molecules in RDKit's Mol format
    """

    # Fix seed
    set_seed(seed)

    # # The gene categories in the control group and the experimental group need to be the same.
    if control.columns.tolist() != pert.columns.tolist():
        raise ValueError("The column names of the two CSV files do not match.")

    data_prepare= dataPrepare(genetic_vocab_file=args.genetic_vocab_file, batch_size=args.batch_size)
    data_loader = data_prepare.main(control=control, pert=pert)

    # Inference
    drugModel.eval()
    model.eval()

    hit_like_smiles = [[] for _ in range(beam_width)]
    hit_like_mols = [[] for _ in range(beam_width)]
    with torch.no_grad():
        for batch, batch_data in enumerate(data_loader):

            input_gene_ids = batch_data["genes"].cuda()
            input_values = batch_data["values"].cuda().float()
            src_key_padding_mask = torch.zeros_like(input_gene_ids, dtype=torch.bool).cuda()

            seq_length = input_values.shape[-1]
            with torch.cuda.amp.autocast(enabled=True):
                output_ctl, \
                output10_ctl, \
                output100_ctl, \
                output1000_ctl = Genetic_Encoder(model=model,
                                                 input_values=input_values[:, :seq_length // 2],
                                                 input_gene_ids=input_gene_ids[:, :seq_length // 2],
                                                 src_key_padding_mask=src_key_padding_mask[:, :seq_length // 2])

                output_desir, \
                output10_desir, \
                output100_desir, \
                output1000_desir = Genetic_Encoder(model=model,
                                                   input_values=input_values[:, seq_length // 2:],
                                                   input_gene_ids=input_gene_ids[:, seq_length // 2:],
                                                   src_key_padding_mask=src_key_padding_mask[:, seq_length // 2:])

                cell_feat = output_desir - output_ctl
                prediction = model.drug_decoder.predict(cell_feat, model.drug_embed, beam_width=beam_width)

            continuous_emb = discrete2continue(torch.from_numpy(np.array(prediction)).cuda(), buckets, pca_mean, pca_comp)
            continuous_emb = continuous_emb.float().reshape((beam_width, -1, 64))

            for i in range(beam_width):
                generate_candidates = drugModel.sample(continuous_emb[i], greedy=True)
                hit_like_mols[i].extend(generate_candidates[0])
                hit_like_smiles[i].extend(generate_candidates[1])

            torch.cuda.empty_cache()

    return hit_like_smiles, hit_like_mols


def Searching_local(control,
              pert,
              beam_width=2,
              ref_smiles=None,
              seed=0,
              first_processed=True,
              data_folder=None):
    """

    Args:
        control: Gene expression data from control groups
        pert: Gene expression data from pert groups
        beam_width: The number of candidate molecules to generate
        seed: Different seeds yield different results
        first_processed: Indicates whether preprocessing is needed
                        (the same search molecular dataset can be skipped after being preprocessed once)
        data_folder: The location where it is saved after the first preprocessing

    Returns:
        smiles: The small molecules found
        scores: The corresponding matching score (modal cosine similarity)
    """

    # Fix seed
    set_seed(seed)

    # # The gene categories in the control group and the experimental group need to be the same.
    if control.columns.tolist() != pert.columns.tolist():
        raise ValueError("The column names of the two CSV files do not match.")

    data_prepare= dataPrepare(genetic_vocab_file=args.genetic_vocab_file, batch_size=args.batch_size)
    data_loader = data_prepare.main(control=control, pert=pert)

    # Preprocessed reference molecules
    with open(args.vocab_drug, 'r') as f:
        vocab_drug = [line.strip().split() for line in f]
    vocab_drug = PairVocab(vocab_drug)

    # If there are too many data that exceed 1000*batch_size, save them in multiple files
    if first_processed or not os.path.isdir(data_folder):
        ref_processed = []
        for batch in ref_smiles:
            try:
                tensorize([batch], vocab=vocab_drug)
                ref_processed.append(batch)
            except:
                print(f'Sorry, {batch} can not tokenized...')
                pass

        all_mol_graphs = []
        batches = [ref_processed[i: i + args.batch_size] for i in range(0, len(ref_processed), args.batch_size)]
        for batch in batches:
            all_mol_graphs.append(tensorize(batch, vocab=vocab_drug))

        os.makedirs(data_folder)
        with open(data_folder+'tensors.pkl', 'wb') as f:
            pickle.dump(all_mol_graphs, f, pickle.HIGHEST_PROTOCOL)

    else:
        ref_processed = ref_smiles
        all_mol_graphs = []
        data_files = [fn for fn in os.listdir(data_folder) if fn.endswith('.pkl')]
        for fn in data_files:
            print('Now turn to {}'.format(fn))
            fn = os.path.join(data_folder, fn)
            with open(fn, 'rb') as f:
                batches = pickle.load(f)

            all_mol_graphs.extend(batches)

    # Inference
    drugModel.eval()
    model.eval()

    smiles = []
    search_score = []
    with torch.no_grad():
        for batch, batch_data in enumerate(data_loader):

            input_gene_ids = batch_data["genes"].cuda()
            input_values = batch_data["values"].cuda().float()
            src_key_padding_mask = torch.zeros_like(input_gene_ids, dtype=torch.bool).cuda()

            seq_length = input_values.shape[-1]
            with torch.cuda.amp.autocast(enabled=True):
                output_ctl, \
                output10_ctl, \
                output100_ctl, \
                output1000_ctl = Genetic_Encoder(model=model,
                                                 input_values=input_values[:, :seq_length // 2],
                                                 input_gene_ids=input_gene_ids[:, :seq_length // 2],
                                                 src_key_padding_mask=src_key_padding_mask[:, :seq_length // 2])

                output_desir, \
                output10_desir, \
                output100_desir, \
                output1000_desir = Genetic_Encoder(model=model,
                                                   input_values=input_values[:, seq_length // 2:],
                                                   input_gene_ids=input_gene_ids[:, seq_length // 2:],
                                                   src_key_padding_mask=src_key_padding_mask[:, seq_length // 2:])

                cell_feat = output_desir - output_ctl

                global_similarity = None
                for i, drug in enumerate(all_mol_graphs):
                    # Drug Embedding
                    drug_feat = drugModel.latent(*drug)
                    drug_feat = torch.bucketize(drug_feat, buckets)

                    # Calculate Similarity (query_cell_num, search_drug_num)
                    drug_feat = model.drug_embed(drug_feat)
                    cell_emb = F.normalize(torch.mean(cell_feat, dim=1), dim=-1)
                    drug_emb = F.normalize(torch.mean(drug_feat, dim=1), dim=-1)
                    similarity = torch.matmul(cell_emb, drug_emb.t()) / model.temp

                    if global_similarity is None:
                        global_similarity = similarity
                    else:
                        global_similarity = torch.cat([global_similarity, similarity], dim=1)

                # Search for the most k related drugs
                top_values, top_indices = torch.topk(global_similarity, k=beam_width, largest=True, dim=1)

                # Record results
                top_indices = top_indices.detach().cpu().numpy()
                top_values = top_values.detach().cpu().numpy()
                smiles_list = np.array(ref_processed)
                for i in range(cell_feat.shape[0]):
                    smiles.append(smiles_list[top_indices[i]].tolist())
                    search_score.append(top_values[i].tolist())

    return np.array(smiles), np.array(search_score)


def Searching(control, \
              pert, \
              searching_smiles, \
              ref_smiles=None, \
              method=None, \
              beam_width=20,\
              seed=0):
    """
        Args:
            control: Gene expression data from control groups
            pert: Gene expression data from pert groups
            ref_smiles: (default None) Can further similarity screen the small molecules found
            method: (default None) Indicator for further similarity screening of the found small molecules
            beam_width: The number of candidate molecules to generate
            seed: Different seeds yield different results
    """

    # Load Dataset
    control = pd.read_csv(control.name, index_col=0, nrows=None) # sample_num * gene_num
    pert = pd.read_csv(pert.name, index_col=0, nrows=None) # sample_num * gene_num

    # # The gene categories in the control group and the experimental group need to be the same.
    if control.columns.tolist() != pert.columns.tolist():
        raise ValueError("The column names of the two CSV files do not match.")

    # Read ref_smiles
    if searching_smiles.name.endswith(".txt"):
        searching_smiles = pd.read_csv(searching_smiles.name, header=None, squeeze=True).to_list()
    elif searching_smiles.name.endswith(".csv"):
        searching_smiles = pd.read_csv(searching_smiles.name,squeeze=True, index_col=0).to_list()

    searched_smiles, searched_scores = Searching_local(control=control,
                                pert=pert,
                                ref_smiles=searching_smiles,
                                beam_width=beam_width,
                                seed=seed)

    if ref_smiles is not None:
        # Read ref_smiles
        if ref_smiles.name.endswith(".txt"):
            ref_smiles = pd.read_csv(ref_smiles.name, header=None, squeeze=True).to_list()
        elif ref_smiles.name.endswith(".csv"):
            ref_smiles = pd.read_csv(ref_smiles.name, squeeze=True, index_col=0).to_list()

        # Screen
        searched_smiles = searched_smiles.T.tolist()
        searched_scores = searched_scores.T.tolist()
        assert len(searched_smiles) == beam_width
        result_smiles = []
        query_data = pd.DataFrame()
        for i in range(beam_width):
            temp = np.tile(np.array(ref_smiles), (len(searched_smiles[i]), 1)).T
            simCalculator = molCalSim(ref_smile=searched_smiles[i], gen_smiles=list(temp))
            temp_smiles, temp_scores = simCalculator.main(query_smiles=list(temp), method=method)
            result_smiles.append(temp_smiles)
            query_data[f"result_smiles{i}"] = np.array(temp_scores).astype(float)
        del temp_smiles, temp_scores
        query_data['max_column'] = query_data.idxmax(axis=1)
        query_data['max_value'] = query_data.max(axis=1)

        max_smiles = []
        max_scores_clip = []
        ref_smiles = []
        for index, row in query_data.iterrows():
            max_column = row['max_column']
            max_smiles.append(searched_smiles[eval(f"{max_column[-1]}")][index])
            max_scores_clip.append(searched_scores[eval(f"{max_column[-1]}")][index])
            ref_smiles.append(result_smiles[eval(f"{max_column[-1]}")][index])
        max_scores = query_data['max_value'].tolist()

        sorted_data = sorted(zip(max_scores, ref_smiles, max_smiles,max_scores_clip), reverse=True)
        max_scores, ref_smiles, max_smiles, max_scores_clip = zip(*sorted_data)

        # Output
        hit_like_smiles = pd.DataFrame({'Ref': ref_smiles, \
                                        'Searched_smiles': max_smiles, \
                                        'Scores': np.array(max_scores).astype(float),
                                        'Searched_scores':np.array(max_scores_clip).astype(float)})
        print(max(max_scores))

        hit_like_smiles.to_csv('GexMolGen_output.csv')
        return 'GexMolGen_output.csv'

    else:
        searched_scores = searched_scores.flatten()
        searched_smiles = searched_smiles.flatten()
        index = np.repeat(np.arange(control.shape[0]), beam_width)
        result = pd.DataFrame({'Searched_smiles':searched_smiles,'Searched_scores':searched_scores,'data_index':index})

        result.to_csv('GexMolGen_output.csv')
        return 'GexMolGen_output.csv'


def Standard(control,
              pert,
              beam_width=2,
              seed=0):
    """

    Args:
        control: Gene expression data from control groups
        pert: Gene expression data from pert groups
        gene_name: (optional) Gene categories
        beam_width: The number of candidate molecules to generate
        seed: Different seeds yield different results

    Returns:
        hit_like_smiles: A CSV file containing candidate molecules
        hit_like_figs: Figures showing candidate molecules in PIL.Image format
    """

    # Generate
    hit_like_smiles, hit_like_mols = generator(control=control,
                                               pert=pert,
                                               beam_width=beam_width,
                                               seed=seed)
    # Draw pictures
    hit_like_mols = np.array(hit_like_mols).T
    hit_like_mols = hit_like_mols.tolist()
    molsPerRow = len(hit_like_mols[0]) if len(hit_like_mols[0]) < 5 else 5
    hit_like_figs = [Draw.MolsToGridImage(hit_like_mols[i], molsPerRow=molsPerRow, subImgSize=(600, 600)) \
                     for i in range(len(hit_like_mols))]
    hit_like_figs = [Image.fromarray(np.array(hit_like_figs[i])) for i in range(len(hit_like_mols))]

    # Output
    hit_like_smiles = np.array(hit_like_smiles).T
    hit_like_smiles = pd.DataFrame(hit_like_smiles)

    hit_like_smiles.to_csv('GexMolGen_output.csv')

    return 'GexMolGen_output.csv', hit_like_figs


def Screen(control,
           pert,
           ref_smiles=None,
           method='Morgan',
           beam_width=1,
           seed=0):
    """

    Args:
        control: Gene expression data from control groups
        pert: Gene expression data from knock-out groups
        ref_smiles: Known inhibitor list
        method: The method chosen to calculate similarity: 'Morgan', 'MACCS', 'Fraggle'
        gene_name: (optional) Gene categories
        beam_width: The number of candidate molecules to generate
        seed: Different seeds yield different results

    Returns:
        hit_like_smiles: A CSV file containing the most similar molecules to the reference molecule,
                        along with their similarity scores
        hit_like_figs: Figures showing the reference molecule and the most similar generated molecules
    """

    # Read ref_smiles
    if ref_smiles.name.endswith(".txt"):
        ref_smiles = pd.read_csv(ref_smiles.name, header=None, squeeze=True).to_list()
    elif ref_smiles.name.endswith(".csv"):
        ref_smiles = pd.read_csv(ref_smiles.name, squeeze=True, index_col=0).to_list()

    # Generate
    hit_like_smiles, _ = generator(control=control,
                                               pert=pert,
                                               beam_width=beam_width,
                                               seed=seed)

    # Screen
    assert len(hit_like_smiles) == beam_width
    result_smiles = []
    query_data = pd.DataFrame()
    for i in range(beam_width):
        temp = np.tile(np.array(ref_smiles), (len(hit_like_smiles[i]), 1)).T
        simCalculator = molCalSim(ref_smile=hit_like_smiles[i], gen_smiles=list(temp))
        temp_smiles, temp_scores = simCalculator.main(query_smiles=list(temp), method=method)
        result_smiles.append(temp_smiles)
        query_data[f"result_smiles{i}"] = np.array(temp_scores).astype(float)
    del temp_smiles, temp_scores
    query_data['max_column'] = query_data.idxmax(axis=1)
    query_data['max_value'] = query_data.max(axis=1)

    max_smiles = []
    ref_smiles = []
    for index, row in query_data.iterrows():
        max_column = row['max_column']
        max_smiles.append(hit_like_smiles[eval(f"{max_column[-1]}")][index])
        ref_smiles.append(result_smiles[eval(f"{max_column[-1]}")][index])
    max_scores = query_data['max_value'].tolist()

    sorted_data = sorted(zip(max_scores, ref_smiles, max_smiles), reverse=True)
    max_scores, ref_smiles, max_smiles = zip(*sorted_data)

    # Draw pictures
    hit_like_mols = [Chem.MolFromSmiles(smi) for smi in max_smiles]
    ref_mols = [Chem.MolFromSmiles(smi) for smi in ref_smiles]
    mols_gather = [ref_mols, hit_like_mols]
    mols_gather = np.array(mols_gather).T
    mols_gather = mols_gather.tolist()
    hit_like_figs = [Draw.MolsToGridImage(mols_gather[i], molsPerRow=2, subImgSize=(600, 600)) \
                     for i in range(len(mols_gather))]
    hit_like_figs = [Image.fromarray(np.array(hit_like_figs[i])) for i in range(len(mols_gather))]

    # Output
    hit_like_smiles = pd.DataFrame({'Ref': ref_smiles, \
                                    'Generate': max_smiles, \
                                    'Scores': np.array(max_scores).astype(float)})
    hit_like_smiles.to_csv('GexMolGen_output.csv')

    return 'GexMolGen_output.csv', hit_like_figs