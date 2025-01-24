# GexMolGen 🐰
Have you ever thought about designing personalized drugs based on your own genes? Sounds fascinating, doesn't it? This is something our GexMolGen aim for!
## Introduction
GexMolGen is a model for generating hit-like molecules based on gene expression signatures. The workflow of GexMolGen is shown below:
![Overview](https://github.com/Bunnybeibei/GexMolGen./assets/77224376/4402dedc-122f-4152-995f-9ce89a6f9fde)

We divide the task of generating hit-like molecules from gene expression profiles into four steps:
1. Encoding of gene expression and small molecular data
2. Matching of genetic modality and small molecular modality
3. Transformation from genetic modality to small molecular modality
4. Generation of small molecules

To simplify the process, we use pre-trained models for the encoders in steps 1, 
namely [*scGPT*](https://github.com/bowang-lab/scGPT) and [*hierVAE*](https://github.com/wengong-jin/hgraph2graph). 
Step 2 is introduced to align the genetic and molecular modalities, while step 3 facilitates the transformation from genetic embeddings to molecular ones. These stages are inspired by [*DALL.E*](https://github.com/openai/dall-e) - simple yet effective! Hahaha..

GexMolGen is an attempt to explore the chemical and biological relationships in the drug discovery process using large language models and multimodal techniques. It has high effectiveness in generating results, flexible input, and strong controllability. For further details, please refer to **our paper** [**GexMolGen: Cross-modal Generation of Hit-like Molecules via Large Language Model Encoding of Gene Expression Signatures**](https://doi.org/10.1093/bib/bbae525).

## Installation
We strongly advise that you individually install `RDKit, FlashAttention, PyTorch` on your device. Here are some configurations from our device for reference:

- CUDA == 11.7
- Python == 3.8
- rdkit == 2023.3.2
- flash-attn == 1.0.1
- torch == 1.13.0+cu117
- gradio == 3.40.1

Please be mindful of version compatibility during your actual setup.

Next, you need to pull down [*scGPT*](https://github.com/bowang-lab/scGPT) under this project. Installation is not necessary.

## Model Parameters

If you want to use our model, you can download it from the provided [link](https://drive.google.com/file/d/1uc7f7qzUjX3e7fSvPxYAzado6jY5myec/view?usp=drive_link). This link already includes the pre-trained 'whole-human' version of the scGPT weights, so there's no need for an additional download.

## Demo
To facilitate your use of our model, we have created an interactive interface. After configuring the environment and adjusting some addresses according to your installation path, 
you can simply run `python server.py` in the command line to display the interface.

We currently have two integrated functions: *Standard* and *Screen*.

- *Standard*: This function generates a specified number of drugs based on gene transcription profile data.
- *Screen*: This function allows you to input reference molecules and similarity calculation methods. It will output the generated results in descending order of similarity to the reference molecules
- *Retrieving*: Retrieval of potential small molecules by providing gene expression profiles and the molecular database you want to search. 🆕

We provide experimental data for *AKT2* (*server_test_ctl.csv and server_test_pert.csv*) 
and reference inhibitors (*AKT2_ref.csv*). You can use the *Screen* function to verify the Result 2.3 in our paper.
![demo](https://github.com/Bunnybeibei/GexMolGen/assets/77224376/b037b93a-5653-44fc-8c5a-82ae03dbf6b3)

## To-do-list
- [ ] Upload video version explanation of the demo
- [x] Upload the complete dataset
- [ ] Upload training code

## Data Availability

The data underlying this article are accessible within the article itself and its supplementary online materials.

You can find the curated data at the following link: 

- [Curated Data](https://zenodo.org/records/11100665?token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTcxNDYyNzUzOSwiZXhwIjoxNzk4Njc1MTk5fQ.eyJpZCI6IjM3ZmFkNGU4LWViMDYtNGNkNy1iOTc4LWI0ZTBkMDk2OWI0YyIsImRhdGEiOnt9LCJyYW5kb20iOiJhOGJjZDM5NWFkY2ZiNDAwNjAzYzIwMTg2ODNjYWI2NCJ9.dADyS-0PBsFKr_z1yDdcDnoGoY5PFOSbnYtt6aIz4RLoNxykoIQffAlzQDPbFgqnZJmp7PmjNXPCXHkDMuZHuA).

The raw data can be downloaded from the following sources:

- [CLUE Data](https://clue.io/data/CMap2020#LINCS2020)

- [ChEMBL Database](https://www.ebi.ac.uk/chembl/)

- [EXCAPES Database](https://solr.ideaconsult.net/search/excape/)

## Citing GexMolGen
```bibtex
@article{10.1093/bib/bbae525,
    author = {Cheng, Jiabei and Pan, Xiaoyong and Fang, Yi and Yang, Kaiyuan and Xue, Yiming and Yan, Qingran and Yuan, Ye},
    title = {GexMolGen: cross-modal generation of hit-like molecules via large language model encoding of gene expression signatures},
    journal = {Briefings in Bioinformatics},
    volume = {25},
    number = {6},
    pages = {bbae525},
    year = {2024},
    month = {10},
    issn = {1477-4054},
    doi = {10.1093/bib/bbae525},
}
```
## Acknowledgements
Finally, we would like to express our deepest gratitude to the authors of [*scGPT*](https://github.com/bowang-lab/scGPT) and [*hierVAE*](https://github.com/wengong-jin/hgraph2graph). 
They have not only created excellent work but also made it open source for the benefit of researchers worldwide.

No matter what questions you have, feel free to contact us via email or raise issues on GitHub. We firmly believe that different perspectives help us develop better tools. 😉
