import torch
import os, pickle, copy
from pathlib import Path
import json
import numpy as np
import argparse

from hgraph import *
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List, Tuple, Dict, Union, Optional
from scGPT.scgpt.model import TransformerModel, FlashTransformerEncoderLayer, TransformerEncoder
from scGPT.scgpt.tokenizer.gene_tokenizer import GeneVocab
from scGPT.scgpt import logger


def dict_to_args(argument_dict):
    parser = argparse.ArgumentParser()

    for key, value in argument_dict.items():
        parser.add_argument(f'--{key}', default=value)

    args = parser.parse_args()
    return args


##################### Beam Search ##########################
class beam_search(object):
    def __init__(self, bs, beam_width, atom_size):
        self.bs = bs
        self.beam_width = beam_width
        self.atom_size = atom_size

    def search(self, pred_x, scores_beam, step):
        scores_temp = F.log_softmax(pred_x, dim=-1)
        scores_next = scores_temp + scores_beam[:, None].expand_as(scores_temp)
        assert scores_next.shape == (self.bs * self.beam_width, self.atom_size)
        scores_next = scores_next.view(self.bs, self.beam_width * self.atom_size)
        a, b = scores_next.topk(k=self.beam_width+step, dim=-1, largest=True, sorted=True)
        scores_next, token_next = a[:,step:], b[:,step:]
        next_batch_beam = []
        for bs_idx in range(self.bs):
            next_sent_beam = []  # save scores, token_id, beam_id
            for rank_beam, (token_id, token_score) in enumerate(zip(token_next[bs_idx], scores_next[bs_idx])):
                beam_id = token_id // self.atom_size
                token_id = token_id % self.atom_size
                effective_beam_id = bs_idx * self.beam_width + beam_id
                next_sent_beam.append((token_score, token_id, effective_beam_id))
                if len(next_sent_beam) == self.beam_width:
                    break
            next_batch_beam.extend(next_sent_beam)
        scores_beam = scores_beam.new([x[0] for x in next_batch_beam])
        beam_tokens = scores_beam.new([x[1] for x in next_batch_beam]).long()
        beam_idx = scores_beam.new([x[2] for x in next_batch_beam]).long()
        assert scores_beam.shape == (self.beam_width * self.bs,)
        assert beam_tokens.shape == (self.beam_width * self.bs,)
        assert beam_idx.shape == (self.beam_width * self.bs,)
        return beam_tokens, beam_idx, scores_beam


##################### Facilitator ##########################
class Facilitator(nn.Module):
    """
    Facilitator.

    """

    def __init__(
        self,
        decoder_dim: int = 512,
        dropout: float = 0.2,
        decoder_layers: int = 4
    ) -> None:
        super().__init__()

        self.dim = decoder_dim

        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=decoder_dim, \
                                       nhead=4,\
                                       dim_feedforward=decoder_dim * 4,\
                                       dropout=dropout,
                                       batch_first = True,
                                       norm_first = True,
                                       activation= 'gelu'),
            decoder_layers,
        )
        self._reset_parameters()

        self.start_token = nn.Parameter(torch.randn(size=(1, 1, decoder_dim)))
        # Predict Drug Features
        self.head_drug = nn.Linear(decoder_dim, 1024)


    def forward(
        self, gene_emb: Tensor, batch_drug: Tensor) -> Union[Tensor, Dict[str, Tensor]]:

        # Drug Embedding
        drug_input = torch.cat([torch.mean(gene_emb, dim=1).unsqueeze(dim=1), batch_drug[:, 1:]], dim=1)

        # Transformer Output
        output = self.transformer_decoder(
            drug_input,  # Remove the last token from the target sequence
            gene_emb,
            tgt_mask=self.generate_diag_mask(batch_drug.size(1)).to(batch_drug.device)
        )

        pred_value = self.head_drug(output)

        return pred_value

    def generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def generate_diag_mask(self,size):
        mask = (torch.diag(torch.ones(size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def predict(self, x, drug_emb, beam_width=20, atom_size=1024):

        # Gene Embedding
        input_emb = torch.mean(x,dim=1).unsqueeze(dim=1)

        assert input_emb.shape[1] == 1

        searcher = beam_search(bs=x.shape[0], beam_width=beam_width, atom_size=atom_size)
        scores_beam = torch.zeros(size=(searcher.bs, searcher.beam_width)).to(x.device)

        # First step generate the same result
        scores_beam[:,1:] = -1e9
        scores_beam = scores_beam.view(-1)
        input_emb = input_emb.unsqueeze(dim=1).expand(-1, searcher.beam_width, -1, -1).contiguous().view(searcher.beam_width*searcher.bs,-1,512)
        x = x.unsqueeze(dim=1).expand(-1, searcher.beam_width, -1, -1).contiguous().view(
            searcher.beam_width * searcher.bs, -1, 512)
        result = np.zeros(shape=(searcher.beam_width*searcher.bs,43))
        for i in range(43):

            # Transformer Output
            out = self.transformer_decoder(
                input_emb,  # Remove the last token from the target sequence
                x,
                # tgt_mask=self.generate_square_subsequent_mask(drug_input.size(1)-1).to(drug_input.device)
                tgt_mask=self.generate_diag_mask(input_emb.size(1)).to(input_emb.device)
            )

            pred_x = self.head_drug(out[:,-1,:])

            beam_tokens, beam_idx, scores_beam = searcher.search(pred_x, scores_beam, i)

            input_emb = input_emb[beam_idx,:,:]
            input_emb = torch.cat((input_emb,drug_emb(beam_tokens).unsqueeze(dim=1)),dim=1)

            result[:, i] = beam_tokens.detach().cpu().numpy()

        return list(result.astype(int))


##################### Model_Encoder ##########################
class ModelLoader(object):
    def __init__(self, drugModel_file=None, geneModel_file=None):
        """
        Import model parameters
        Args:
            drugModel_file: The address of drugModel
            geneModer_file: The address of geneModel
        """
        self.drugModel_file = drugModel_file
        self.geneModel_file = geneModel_file

    def GeneLoader(self,args):

        # Basic configuration
        hyperparameter_defaults = dict(
            dataset_name="L1000",  # dataset名字
            load_model=self.geneModel_file,
            nlayers=4,
            nhead=4,
            pad_token="<pad>",
            use_batch=False,  # 是否去批次效应
            fast_transformer=True,  # 是否用fast_transformer
            pre_norm=False,  # transformer是先norm还是后norm
            amp=True,  # Automatic Mixed Precision
            decoder_layer=4,
            n_layers=8,
            decoder_emb=512,
        )

        config = dict_to_args(hyperparameter_defaults)

        special_tokens = [config.pad_token, "<cls>", "<eoc>"]
        if config.load_model is not None:
            model_dir = Path(config.load_model)
            self.genetic_model_config_file = model_dir / "args.json"
            self.genetic_vocab_file = model_dir / "vocab.json"
            self.genetic_model_file = args.geneModel_path
            self.vocab_gene = GeneVocab.from_file(self.genetic_vocab_file)
            for s in special_tokens:
                if s not in self.vocab_gene:
                    vocab.append_token(s)

            with open(self.genetic_model_config_file, "r") as f:
                model_configs = json.load(f)

            embsize = model_configs["embsize"]
            nhead = model_configs["nheads"]
            d_hid = model_configs["d_hid"]
        ntokens = len(self.vocab_gene)

        # Importing model
        model = TransformerModel(
            ntokens,
            embsize,
            nhead,
            d_hid,
            nlayers=config.n_layers-args.granu-1 if args.granu > 0 else config.n_layers,
            vocab=self.vocab_gene,
            dropout=args.dropout,
            pad_token="<pad>",
            pad_value=-2,
            do_dab=False,
            use_batch_labels=config.use_batch,
            num_batch_labels=3,
            use_fast_transformer=True,
            pre_norm=config.pre_norm,
            domain_spec_batchnorm = "batchnorm",
        )


        # Importing model parameters
        if config.load_model is not None:
            try:
                model.load_state_dict(torch.load(self.genetic_model_file, map_location='cuda'))
                print(f"Loading all model params from {self.genetic_model_file}")
            except:
                pretrained_dict_orig = torch.load(self.genetic_model_file)
                pretrained_dict = copy.deepcopy(pretrained_dict_orig)

                encoder_layers = FlashTransformerEncoderLayer(
                    embsize,
                    nhead,
                    d_hid,
                    args.dropout,
                    batch_first=True,
                )

                if args.granu == 0:
                    pass
                elif args.granu == 1:
                    for key, value in pretrained_dict_orig.items():
                        if 'transformer_encoder.layers.7' in key:
                            pretrained_dict[
                                key.replace('transformer_encoder.layers.7', 'transformer_encoder_contrast.layers.0')] = value
                        elif 'transformer_encoder.layers.6' in key:
                            pretrained_dict[
                                key.replace('transformer_encoder.layers.6', 'transformer_encoder10.layers.0')] = value
                            model.transformer_encoder10 = TransformerEncoder(encoder_layers, 1)
                            model.transformer_encoder_contrast = TransformerEncoder(encoder_layers, 1)
                            model.cls_decoder10 = nn.Sequential(
                                nn.Linear(embsize, embsize),
                                nn.LeakyReLU(),
                                nn.Dropout(p=args.dropout),
                                nn.Linear(embsize, 10),
                                nn.LeakyReLU(),
                                nn.Dropout(p=args.dropout))
                elif args.granu == 2:
                    for key, value in pretrained_dict_orig.items():
                        if 'transformer_encoder.layers.7' in key:
                            pretrained_dict[
                                key.replace('transformer_encoder.layers.7', 'transformer_encoder_contrast.layers.0')] = value
                        elif 'transformer_encoder.layers.6' in key:
                            pretrained_dict[
                                key.replace('transformer_encoder.layers.7', 'transformer_encoder100.layers.0')] = value
                        elif 'transformer_encoder.layers.5' in key:
                            pretrained_dict[
                                key.replace('transformer_encoder.layers.6', 'transformer_encoder10.layers.0')] = value
                        model.transformer_encoder10 = TransformerEncoder(encoder_layers, 1)
                        model.transformer_encoder100 = TransformerEncoder(encoder_layers, 1)
                        model.transformer_encoder_contrast = TransformerEncoder(encoder_layers, 1)
                        model.cls_decoder10 = nn.Sequential(
                            nn.Linear(embsize, embsize),
                            nn.LeakyReLU(),
                            nn.Dropout(p=args.dropout),
                            nn.Linear(embsize, 10),
                            nn.LeakyReLU(),
                            nn.Dropout(p=args.dropout))
                        model.cls_decoder100 = nn.Sequential(
                            nn.Linear(embsize, embsize),
                            nn.LeakyReLU(),
                            nn.Dropout(p=args.dropout),
                            nn.Linear(embsize, 100),
                            nn.LeakyReLU(),
                            nn.Dropout(p=args.dropout))
                elif args.granu == 3:
                    for key, value in pretrained_dict_orig.items():
                        if 'transformer_encoder.layers.7' in key:
                            pretrained_dict[
                                key.replace('transformer_encoder.layers.7', 'transformer_encoder_contrast.layers.0')] = value
                        elif 'transformer_encoder.layers.6' in key:
                            pretrained_dict[
                                key.replace('transformer_encoder.layers.6', 'transformer_encoder1000.layers.0')] = value
                        elif 'transformer_encoder.layers.5' in key:
                            pretrained_dict[
                                key.replace('transformer_encoder.layers.5', 'transformer_encoder100.layers.0')] = value
                        elif 'transformer_encoder.layers.4' in key:
                            pretrained_dict[
                                key.replace('transformer_encoder.layers.4', 'transformer_encoder10.layers.0')] = value
                        model.transformer_encoder10 = TransformerEncoder(encoder_layers, 1)
                        model.transformer_encoder100 = TransformerEncoder(encoder_layers, 1)
                        model.transformer_encoder1000 = TransformerEncoder(encoder_layers, 1)
                        model.transformer_encoder_contrast = TransformerEncoder(encoder_layers, 1)
                        model.cls_decoder10 = nn.Sequential(
                            nn.Linear(embsize, embsize),
                            nn.LeakyReLU(),
                            nn.Dropout(p=args.dropout),
                            nn.Linear(embsize, 10),
                            nn.LeakyReLU(),
                            nn.Dropout(p=args.dropout))
                        model.cls_decoder100 = nn.Sequential(
                            nn.Linear(embsize, embsize),
                            nn.LeakyReLU(),
                            nn.Dropout(p=args.dropout),
                            nn.Linear(embsize, 100),
                            nn.LeakyReLU(),
                            nn.Dropout(p=args.dropout))
                        model.cls_decoder1000 = nn.Sequential(
                            nn.Linear(embsize, embsize),
                            nn.LeakyReLU(),
                            nn.Dropout(p=args.dropout),
                            nn.Linear(embsize, 1000),
                            nn.LeakyReLU(),
                            nn.Dropout(p=args.dropout))

                model.drug_embed = nn.Embedding(1024, embsize)
                model.temp = nn.Parameter(torch.ones([]) * 0.2)

                model_dict = model.state_dict()
                for k, v in pretrained_dict.items():
                    if k not in model_dict or v.shape != model_dict[k].shape:
                        print(f'Don\'t match! {k}:{v.shape}')

                pretrained_dict = {
                    k: v
                    for k, v in pretrained_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }

                for k, v in pretrained_dict.items():
                    logger.info(f"Loading params {k} with shape {v.shape}")
                model_dict.update(pretrained_dict)
                model.init_weights()
                model.load_state_dict(model_dict)

        return model

    def DrugLoader(self, drug_vocab_file):

        # Basic configuration
        parser = argparse.ArgumentParser()

        parser.add_argument('--vocab_drug', default=drug_vocab_file)
        parser.add_argument('--atom_vocab', default=common_atom_vocab)
        parser.add_argument('--rnn_type', type=str, default='LSTM')
        parser.add_argument('--hidden_size', type=int, default=250)
        parser.add_argument('--embed_size', type=int, default=250)
        parser.add_argument('--batch_size', type=int, default=50)
        parser.add_argument('--latent_size', type=int, default=64)
        parser.add_argument('--depthT', type=int, default=15)
        parser.add_argument('--depthG', type=int, default=15)
        parser.add_argument('--diterT', type=int, default=1)
        parser.add_argument('--diterG', type=int, default=3)
        parser.add_argument('--dropout_drug', type=float, default=0.0)

        parser.add_argument('--bs', type=int, default=256)
        parser.add_argument('--beam_width', type=int, default=20)
        args = parser.parse_args()

        vocab = [x.strip("\r\n ").split() for x in open(args.vocab_drug)]
        args.vocab_drug = PairVocab(vocab)

        # Importing model
        drugModel = HierVAE(args)
        drugModel.R_mean2 = torch.nn.Linear(args.latent_size, args.latent_size)
        drugModel.R_var2 = torch.nn.Linear(args.latent_size, args.latent_size)

        # Importing model parameters
        if self.drugModel_file is not None:
            try:
                drugModel.load_state_dict(torch.load(self.drugModel_file,map_location='cuda'))
                print(f"Loading all model params from {self.drugModel_file}")
            except:
                model_dict = drugModel.state_dict()
                pretrained_dict = torch.load(self.drugModel_file,map_location='cuda')[0]
                pretrained_dict = {
                    k: v
                    for k, v in pretrained_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                for k, v in pretrained_dict.items():
                    print(f"Loading params {k} with shape {v.shape}")
                model_dict.update(pretrained_dict)
                drugModel.load_state_dict(model_dict)

        return drugModel

    def FaciLoader(self, args, path=None):

        with open(self.genetic_model_config_file, "r") as f:
            model_configs = json.load(f)

        embsize = model_configs["embsize"]

        model = Facilitator(dropout=args.dropout, decoder_dim=embsize, decoder_layers=args.faci_n_layers)

        if path != None:
            model.load_state_dict(torch.load(path, map_location='cuda'))

        return model


##################### Genetic_Encoder ##########################
def Genetic_Encoder(model, \
                    input_gene_ids, \
                    input_values,\
                    src_key_padding_mask):

    src = model.encoder(input_gene_ids)
    values = model.value_encoder(input_values)
    total_embs = src + values

    total_embs = model.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

    output = model.transformer_encoder_contrast(
        total_embs, src_key_padding_mask=src_key_padding_mask
    )
    return output
