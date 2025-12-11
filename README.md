# Contextual and Combinatorial Structure in Sperm Whale Vocalisations

This repository contains code for the paper _"Contextual and Combinatorial Structure in Sperm Whale Vocalisations,"_ by Pratyusha Sharma, Shane Gero, Roger Payne, David Gruber, Daniela Rus*, Antonio Torralba*, Jacob Andreas*. 


## Abstract

Sperm whales (_Physeter macrocephalus_) are highly social mammals that communicate using sequences of clicks called codas. While a subset of codas have been shown to encode information about caller identity, almost everything else about the sperm whale communication system, its structure, and information-carrying capacity, remains unknown. We show that codas exhibit contextual and combinatorial structure. First, we report previously undescribed modulations of coda structure that are sensitive to the conversational context in which they occur. We call these rubato and ornamentation. These are systematically controlled and imitated across whales. Second, we show that codas form a combinatorial coding system in which rubato and ornamentation combine with two context-independent features we call rhythm and tempo to produce a large inventory of distinguishable codas. Sperm whale vocalisations are more expressive and structured than previously believed, and built from a repertoire comprising nearly an order of magnitude more distinguishable codas. These results show context-sensitive and combinatorial vocalisation extends beyond humans, and can appear in organisms with divergent evolutionary lineage and vocal apparatus.

## Neural pipeline status

- The pickled rhythm/ornament/tempo artifacts plus the raw dialogue CSVs already provide fixed-length feature vectors, context windows, and labels needed for a learning system. `src/data_pipeline.py` centralizes all loaders, padding, normalization, and context-window building so any downstream training or analysis code uses the same preprocessing.
- `src/model.py` defines a simple context-encoder architecture that turns a window of previous codas into both a regression prediction for the next coda and an embedding for future contrastive objectives.
- `train/train.py` provides a CLI entrypoint that loads the dialogues, wraps them in a PyTorch `Dataset`, trains the model with mean-squared error, and saves the fingerprinted weights inside `artifacts/coda_model.pt`. Run it with `python3 train/train.py --epochs 10 --batch 32` after installing dependencies (`pip install torch`).
