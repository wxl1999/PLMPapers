# PLM papers

![](https://img.shields.io/github/last-commit/RUCAIBox/PLMPapers?color=blue) ![](https://img.shields.io/badge/PaperNumber-285-brightgreen) ![](https://img.shields.io/badge/PRs-Welcome-red)

> Contributed by [Xiaolei Wang](https://github.com/wxl1999)

Large-scale pre-trained language models (PLMs) such as BERT and GPT have achieved great success and become a milestone in NLP.

In this repo, we collect some representative PLM papers in recent years based on the number of citations and papers in 2021 from top conferences (e.g., ACL, EMNLP, ICLR, ICML, NeurIPS).

We will keep the repo updated and welcome pull requests and issues! Thanks for your stars and forks!

**Table of Contents**
- [Survey](#survey)
- [Benchmark](#benchmark)
- [PLM Design](#plm-design)
  - [General](#general)
  - [Knowledge](#knowledge)
  - [Multilingual](#multilingual)
  - [Multi-Modal](#multi-modal)
  - [Information Retrieval](#information-retrieval)
  - [Code](#code)
  - [Others](#others)
- [PLM Analysis](#plm-analysis)
  - [Knowledge](#knowledge-1)
  - [Robustness](#robustness)
  - [Sparsity](#sparsity)
  - [Others](#others-1)
- [Efficient PLM](#efficient-plm)
  - [Training](#training)
  - [Inference](#inference)
  - [Compression](#compression)
- [PLM Adaptation](#plm-adaptation)
  - [Two-Stage](#two-stage)
  - [Multi-Task](#multi-task)
  - [Adapater](#adapater)
  - [Prompt](#prompt)
  - [Others](#others-2)

## Survey

1. "Pre-trained models for natural language processing: A survey". `Science China Technological Sciences(2020)` [[PDF]](https://www.sciengine.com/publisher/scp/journal/SCTS/63/10/10.1007/s11431-020-1647-3?slug=fulltext)
2. "Which *BERT? A Survey Organizing Contextualized Encoders". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.emnlp-main.608.pdf)
3. "A Primer in BERTology: What We Know About How BERT Works". `TACL(2020)` [[PDF]](https://aclanthology.org/2020.tacl-1.54.pdf)
4. "From static to dynamic word representations: a survey". `International Journal of Machine Learning and Cybernetics(2020)` [[PDF]](http://ir.hit.edu.cn/~car/papers/icmlc2020-wang.pdf)
5. "Overview of the Transformer-based Models for NLP Tasks". `2020 15th Conference on Computer Science and Information Systems (FedCSIS)` [[PDF]](https://ieeexplore.ieee.org/abstract/document/9222960)
6. "A Survey on Contextual Embeddings". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2003.07278)
7. "The NLP Cookbook: Modern Recipes for Transformer Based Deep Learning Architectures". `IEEE Access(2021)` [[PDF]](https://ieeexplore.ieee.org/abstract/document/9422763)
8. "Pre-Trained Models: Past, Present and Future". `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2106.07139)
9.  "Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing". `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2107.13586)
10. "AMMUS : A Survey of Transformer-based Pretrained Models in Natural Language Processing". `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2108.05542)
11. "On the Opportunities and Risks of Foundation Models". `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2108.07258)
12. "Paradigm Shift in Natural Language Processing". `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2109.12575)
13. "Recent Advances in Natural Language Processing via Large Pre-Trained Language Models: A Survey". `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2111.01243)

## Benchmark

1. **XNLI**: "XNLI: Evaluating Cross-lingual Sentence Representations". `EMNLP(2018)` [[PDF]](https://aclanthology.org/D18-1269.pdf) [[Dataset]](https://github.com/facebookresearch/XNLI)
2. **GLUE**: "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding". `ICLR(2019)` [[Homepage]](https://gluebenchmark.com/)
3. **SuperGLUE**: "SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems". `NeurIPS(2019)` [[Homepage]](https://super.gluebenchmark.com/)
4. **CLUE**: "CLUE: A Chinese Language Understanding Evaluation Benchmark". `COLING(2020)` [[Homepage]](https://www.cluebenchmarks.com/)
5. **XTREME**: "XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization". `ICML(2020)` [[Homepage]](https://sites.research.google/xtreme)
6. **XGLUE**: "XGLUE: A New Benchmark Dataset for Cross-lingual Pre-training, Understanding and Generation". `EMNLP(2020)` [[Homepage]](https://microsoft.github.io/XGLUE/)
7. **DialoGLUE**: "DialoGLUE: A Natural Language Understanding Benchmark for Task-Oriented Dialogue". `arXiv(2020)` [[Homepage]](https://eval.ai/web/challenges/challenge-page/708/overview)

## PLM Design

### General

1. **GPT**: "Improving Language Understanding by Generative Pre-Training". `OpenAI(2018)` [[Project]](https://openai.com/blog/language-unsupervised/)
2. **GPT-2**: "Language Models are Unsupervised Multitask Learners". `OpenAI(2019)` [[Project]](https://openai.com/blog/better-language-models/)
3. **BERT**: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". `NAACL(2019)` [[PDF]](https://arxiv.org/pdf/1810.04805.pdf) [[Code]](https://github.com/google-research/bert)
4. **XLNet**: "XLNet: Generalized Autoregressive Pretraining for Language Understanding". `NeurIPS(2019)` [[PDF]](https://proceedings.neurips.cc/paper/2019/file/dc6a7e655d7e5840e66733e9ee67cc69-Paper.pdf) [[Code]](https://github.com/zihangdai/xlnet)
5. **SBERT**: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks". `ACL(2019)` [[PDF]](https://aclanthology.org/D19-1410.pdf) [[Code]](https://github.com/UKPLab/sentence-transformers)
6. **UniLM**: "Unified Language Model Pre-training for Natural Language Understanding and Generation". `NeurIPS(2019)` [[PDF]](https://proceedings.neurips.cc/paper/2019/file/c20bb2d9a50d5ac1f713f8b34d9aac5a-Paper.pdf) [[Code]](https://github.com/microsoft/unilm)
7. **MASS**: "MASS: Masked Sequence to Sequence Pre-training for Language Generation". `ICML(2019)` [[PDF]](http://proceedings.mlr.press/v97/song19d/song19d.pdf) [[Code]](https://github.com/microsoft/MASS)
8.  **Chinese-BERT-wwm**: "Pre-Training with Whole Word Masking for Chinese BERT". `arXiv(2019)` [[PDF]](https://arxiv.org/pdf/1906.08101.pdf) [[Code]](https://github.com/ymcui/Chinese-BERT-wwm)
9.  "Cloze-driven Pretraining of Self-attention Networks". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1539.pdf)
10. "BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model". `Workshop on Methods for Optimizing and Evaluating Neural Language Generation(2019)` [[PDF]](https://aclanthology.org/W19-2304.pdf) [[Code]](https://github.com/nyu-dl/bert-gen)
11. **GPT-3**: "Language Models are Few-Shot Learners". `NeurIPS(2020)` [[PDF]](https://papers.nips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) [[Code]](https://github.com/openai/gpt-3)
12. **T5**: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer". `JMLR(2020)` [[PDF]](https://jmlr.org/papers/volume21/20-074/20-074.pdf) [[Code]](https://github.com/google-research/text-to-text-transfer-transformer)
13. **BART**: "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension". `ACL(2020)` [[PDF]](https://aclanthology.org/2020.acl-main.703.pdf) [[Code]](https://github.com/pytorch/fairseq)
14. **Poly-encoders**: "Poly-encoders: Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring". `ICLR(2020)` [[PDF]](https://openreview.net/pdf?id=SkxgnnNFvH)
15. **SpanBERT**: "SpanBERT: Improving Pre-training by Representing and Predicting Spans". `TACL(2020)` [[PDF]](https://aclanthology.org/2020.tacl-1.5.pdf) [[Code]](https://github.com/facebookresearch/SpanBERT)
16. **ERNIE 2.0**: "ERNIE 2.0: A Continual Pre-Training Framework for Language Understanding". `AAAI(2020)` [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/6428/6284) [[Code]](https://github.com/PaddlePaddle/ERNIE)
17. **SemBERT**: "Semantics-Aware BERT for Language Understanding". `AAAI(2020)` [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/6510/6366) [[Code]](https://github.com/cooelf/)
18. "Leveraging Pre-trained Checkpoints for Sequence Generation Tasks". `TACL(2020)` [[PDF]](https://aclanthology.org/2020.tacl-1.18.pdf) [[Code]](https://github.com/google-research/google-research/tree/master/bertseq2seq)
19. **ProphetNet**: "ProphetNet: Predicting Future N-gram for Sequence-to-SequencePre-training". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.findings-emnlp.217.pdf)
20. **UniLMv2**: "UniLMv2: Pseudo-Masked Language Models for Unified Language Model Pre-Training". `ICML(2020)` [[PDF]](http://proceedings.mlr.press/v119/bao20a/bao20a.pdf) [[Code]](https://github.com/microsoft/unilm)
21. **MacBERT**: "Revisiting Pre-Trained Models for Chinese Natural Language Processing". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.findings-emnlp.58.pdf) [[Code]](https://github.com/ymcui/MacBERT)
22. **MPNet**: "MPNet: Masked and Permuted Pre-training for Language Understanding". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2004.09297) [[Code]](https://github.com/microsoft/MPNet)
23. **DEBERTA**: "DeBERTa: Decoding-enhanced BERT with Disentangled Attention". `ICLR(2021)` [[PDF]](https://openreview.net/pdf?id=XPZIaotutsD) [[Code]](https://github.com/microsoft/DeBERTa)
24. **PALM**: "PALM: Pre-training an Autoencoding&Autoregressive Language Model for Context-conditioned Generation". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.emnlp-main.700.pdf)
25. **Optimus**: "Optimus: Organizing Sentences via Pre-trained Modeling of a Latent Space". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.emnlp-main.378.pdf) [[Code]](https://github.com/ChunyuanLI/Optimus)
26. "Self-training Improves Pre-training for Natural Language Understanding". `NAACL(2021)` [[PDF]](https://aclanthology.org/2021.naacl-main.426.pdf) [[Code]](https://github.com/facebookresearch/SentAugment)
27. **CAPT**: "Rethinking Denoised Auto-Encoding in Language Pre-Training". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.232.pdf)
28. "Frustratingly Simple Pretraining Alternatives to Masked Language Modeling". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.249.pdf) [[Code]](https://github.com/gucci-j/light-transformer-emnlp2021)
29. "Convolutions and Self-Attention: Re-interpreting Relative Positions in Pre-trained Language Models". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.333.pdf) [[Code]](https://github.com/mlpc-ucsd/BERT_Convolutions)
30. **ERNIE-Doc**: "ERNIE-Doc: A Retrospective Long-Document Modeling Transformer". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.227.pdf) [[Code]](https://github.com/PaddlePaddle/ERNIE/tree/repro/ernie-doc)
31. "Pre-training Universal Language Representation". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.398.pdf) [[Code]](https://github.com/Liyianan/ULR)

### Knowledge

1. **ERNIE(Baidu)**: "ERNIE: Enhanced Representation through Knowledge Integration". `arXiv(2019)` [[PDF]](https://arxiv.org/pdf/1904.09223) [[Code]](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE)
2. **KnowBert**: "Knowledge Enhanced Contextual Word Representations". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1005.pdf)
3. **ERNIE(Tsinghua)**: "ERNIE: Enhanced Language Representation with Informative Entities". `ACL(2019)` [[PDF]](https://aclanthology.org/P19-1139.pdf) [[Code]](https://github.com/thunlp/ERNIE)
4. **COMET**: "COMET: Commonsense Transformers for Automatic Knowledge Graph Construction". `ACL(2019)` [[PDF]](https://aclanthology.org/P19-1470.pdf) [[Code]](https://github.com/atcbosselut/comet-commonsense)
5. **K-BERT**: "K-BERT: Enabling Language Representation with Knowledge Graph". `AAAI(2020)` [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/5681/5537) [[Code]](https://github.com/autoliuweijie/K-BERT)
6. **WKLM**: "Pretrained Encyclopedia: Weakly Supervised Knowledge-Pretrained Language Model". `ICLR(2020)` [[PDF]](https://openreview.net/pdf?id=BJlzm64tDH)
7.  **LUKE**: "LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.emnlp-main.523.pdf) [[Code]](https://github.com/studio-ousia/luke)
8.  **K-Adapter**: "K-Adapter: Infusing Knowledge into Pre-Trained Models with Adapters". `ICLR(2021)` [[PDF]](https://openreview.net/pdf?id=CLnj31GZ4cI)
9.  **KEPLER**: "KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation". `TACL(2021)` [[PDF]](https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl_a_00360/1923927/tacl_a_00360.pdf) [[Code]](https://github.com/THU-KEG/KEPLER)
10. **RuleBERT**: "RuleBERT: Teaching Soft Rules to Pre-Trained Language Models". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.110.pdf) [[Code]](http://github.com/MhmdSaiid/RuleBert)
11. **BeliefBank**: "Exploring the Role of BERT Token Representations to Explain Sentence Probing Results". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.697.pdf) [[Code]](https://allenai.org/data/beliefbank)
12. **Phrase-BERT**: "Phrase-BERT: Improved Phrase Embeddings from BERT with an Application to Corpus Exploration". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.846.pdf) [[Code]](https://github.com/sf-wa-326/phrase-bert-topic-model)
13. "Syntax-Enhanced Pre-trained Model". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.420.pdf) [[Code]](https://github.com/Hi-ZenanXu/Syntax-Enhanced)
14. **StructFormer**: "StructFormer: Joint Unsupervised Induction of Dependency and Constituency Structure from Masked Language Modeling". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.559.pdf)
15. **ERICA**: "ERICA: Improving Entity and Relation Understanding for Pre-trained Language Models via Contrastive Learning". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.260.pdf) [[Code]](https://github.com/thunlp/ERICA)
16. "Structural Guidance for Transformer Language Models". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.289.pdf) [[Code]](https://github.com/IBM/transformers-struct-guidance)
17. **HORNET**: "HORNET: Enriching Pre-trained Language Representations with Heterogeneous Knowledge Sources". `CIKM(2021)` [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3459637.3482436)
18. "Drop Redundant, Shrink Irrelevant: Selective Knowledge Injection for Language Pretraining". `IJCAI(2021)` [[PDF]](https://www.ijcai.org/proceedings/2021/0552.pdf)

### Multilingual

1. **XLM**: "Cross-lingual Language Model Pretraining". `arXiv(2019)` [[PDF]](https://arxiv.org/pdf/1901.07291) [[Code]](https://github.com/facebookresearch/XLM)
2. "Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond". `TACL(2019)` [[PDF]](https://aclanthology.org/Q19-1038.pdf) [[Code]](https://github.com/facebookresearch/LASER)
3. **UDify**: "75 Languages, 1 Model: Parsing Universal Dependencies Universally". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1279.pdf) [[Code]](https://github.com/hyperparticle/udify)
4. **Unicoder**: "Unicoder: A Universal Language Encoder by Pre-training with Multiple Cross-lingual Tasks". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1252.pdf)
5. **XLM-R**: "Unsupervised Cross-lingual Representation Learning at Scale". `ACL(2020)` [[PDF]](https://aclanthology.org/2020.acl-main.747.pdf)
6. "Multilingual Alignment of Contextual Word Representations". `ICLR(2020)` [[PDF]](https://openreview.net/pdf?id=r1xCMyBtPS)
7. **mBART**: "Multilingual Denoising Pre-training for Neural Machine Translation". `TACL(2020)` [[PDF]](https://aclanthology.org/2020.tacl-1.47.pdf) [[Code]](https://github.com/pytorch/fairseq/tree/master/examples/mbart)
8. **mT5**: "mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer". `NAACL(2021)` [[PDF]](https://aclanthology.org/2021.naacl-main.41.pdf) [[Code]](https://goo.gle/mt5-code)
9. **InfoXLM**: "InfoXLM: An Information-Theoretic Framework for Cross-Lingual Language Model Pre-Training". `NAACL(2021)` [[PDF]](https://aclanthology.org/2021.naacl-main.280.pdf) [[Code]](https://aka.ms/infoxlm)
10. "Allocating Large Vocabulary Capacity for Cross-lingual Language Model Pre-training". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.257.pdf) [[Code]](https://github.com/bozheng-hit/VoCapXLM)
11. **ERNIE-M**: "ERNIE-M: Enhanced Multilingual Representation by Aligning Cross-lingual Semantics with Monolingual Corpora". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.3.pdf) [[Code]](https://github.com/PaddlePaddle/ERNIE)
12. "A Simple Geometric Method for Cross-Lingual Linguistic Transformations with Pre-trained Autoencoders". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.792.pdf)
13. "Boosting Cross-Lingual Transfer via Self-Learning with Uncertainty Estimation". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.538.pdf)
14. "How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.243.pdf) [[Code]](https://github.com/Adapter-Hub/hgiyt)
15. "Multilingual Pre-training with Universal Dependency Learning". `NeurIPS(2021)` [[PDF]](https://proceedings.neurips.cc/paper/2021/file/473803f0f2ebd77d83ee60daaa61f381-Paper.pdf)

### Multi-Modal

1. **ViLBERT**: "ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks". `NeuralIPS(2019)` [[PDF]](https://proceedings.neurips.cc/paper/2019/file/c74d97b01eae257e44aa9d5bade97baf-Paper.pdf)
2. **LXMERT**: "LXMERT: Learning Cross-Modality Encoder Representations from Transformers". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1514.pdf) [[Code]](https://github.com/airsplay/lxmert)
3. **VideoBERT**: "VideoBERT: A Joint Model for Video and Language Representation Learning" `ICCV(2019)` [[PDF]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Sun_VideoBERT_A_Joint_Model_for_Video_and_Language_Representation_Learning_ICCV_2019_paper.pdf)
4. **VisualBERT**: "VisualBERT: A Simple and Performant Baseline for Vision and Language". `arXiv(2019)` [[PDF]](https://arxiv.org/pdf/1908.03557.pdf)
5. **B2T2**: "Fusion of Detected Objects in Text for Visual Question Answering". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1219.pdf) [[Code]](https://github.com/google-research/language/tree/master/language/question_answering/b2t2)
6. **VL-BERT**: "VL-BERT: Pre-training of Generic Visual-Linguistic Representations". `ICLR(2020)` [[PDF]](https://openreview.net/pdf?id=SygXPaEYvH) [[Code]](https://github.com/jackroos/VL-BERT)
7. **Unicoder-VL**: "Unicoder-VL: A Universal Encoder for Vision and Language by Cross-Modal Pre-Training". `AAAI(2020)` [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/6795/6649)
8. **VLP**: "Unified Vision-Language Pre-Training for Image Captioning and VQA". `AAAI(2020)` [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/7005/6859) [[Code]](https://github.com/LuoweiZhou/VLP)
9.  **UNITER**: "UNITER: UNiversal Image-TExt Representation Learning". `ECCV(2020)` [[PDF]](https://arxiv.org/pdf/1909.11740) [[Code]](https://github.com/ChenRocks/UNITER)
10. **Oscar**: "Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks". `ECCV(2020)` [[PDF]](https://arxiv.org/pdf/2004.06165) [[Code]](https://github.com/microsoft/Oscar)
11. "12-in-1: Multi-Task Vision and Language Representation Learning". `CVPR(2020)` [[PDF]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lu_12-in-1_Multi-Task_Vision_and_Language_Representation_Learning_CVPR_2020_paper.pdf) [[Code]](https://github.com/facebookresearch/vilbert-multi-task)
12. **ActBERT**: "ActBERT: Learning Global-Local Video-Text Representations". `CVPR(2020)` [[PDF]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhu_ActBERT_Learning_Global-Local_Video-Text_Representations_CVPR_2020_paper.pdf)
13. **VLN**: "Vision-Language Navigation With Self-Supervised Auxiliary Reasoning Tasks". `CVPR(2020)` [[PDF]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhu_Vision-Language_Navigation_With_Self-Supervised_Auxiliary_Reasoning_Tasks_CVPR_2020_paper.pdf)
14. **VILLA**: "Large-Scale Adversarial Training for Vision-and-Language Representation Learning". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2006.06195.pdf) [[Code]](https://github.com/zhegan27/VILLA)
15. **ImageBERT**: "ImageBERT: Cross-modal Pre-training with Large-scale Weak-supervised Image-Text Data". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2001.07966.pdf)
16. **ALIGN**: "Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision". `ICML(2021)` [[PDF]](https://arxiv.org/pdf/2102.05918.pdf) 
17. **ClipBERT**: "Less Is More: ClipBERT for Video-and-Language Learning via Sparse Sampling". `CVPR(2021)` [[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Lei_Less_Is_More_ClipBERT_for_Video-and-Language_Learning_via_Sparse_Sampling_CVPR_2021_paper.pdf) [[Code]](https://github.com/jayleicn/ClipBERT)
18. **DALL·E**: "Zero-Shot Text-to-Image Generation". `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2102.12092.pdf) [[Code]](https://github.com/openai/DALL-E)
19. **CLIP**: "Learning Transferable Visual Models From Natural Language Supervision". `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2103.00020.pdf) [[Code]](https://github.com/OpenAI/CLIP)
20. **IPT**: "Pre-Trained Image Processing Transformer". `CVPR(2021)` [[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Pre-Trained_Image_Processing_Transformer_CVPR_2021_paper.pdf) [[Code]](https://github.com/huawei-noah/Pretrained-IPT)
21. **CvT**: "CvT: Introducing Convolutions to Vision Transformers". `ICCV(2021)` [[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_CvT_Introducing_Convolutions_to_Vision_Transformers_ICCV_2021_paper.pdf) [[Code]](https://github.com/leoxiaobin/CvT)
22. "Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision". `ICML(2021)` [[PDF]](http://proceedings.mlr.press/v139/jia21b/jia21b.pdf)
23. **TERA**: "TERA: Self-Supervised Learning of Transformer Encoder Representation for Speech". `TASLP(2021)` [[PDF]](https://ieeexplore.ieee.org/abstract/document/9478264) [[Code]](https://github.com/s3prl/s3prl)
24. **CaiT**: "Going deeper with Image Transformers". `ICCV(2021)` [[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Touvron_Going_Deeper_With_Image_Transformers_ICCV_2021_paper.pdf) [[Code]](https://github.com/facebookresearch/deit)
25. **ViViT**: "ViViT: A Video Vision Transformer". `ICCV(2021)` [[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Arnab_ViViT_A_Video_Vision_Transformer_ICCV_2021_paper.pdf) [[Code]](https://github.com/google-research/scenic)
26. **VirTex**: "VirTex: Learning Visual Representations From Textual Annotations". `CVPR(2021)` [[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Desai_VirTex_Learning_Visual_Representations_From_Textual_Annotations_CVPR_2021_paper.pdf) [[Code]](https://github.com/kdexd/virtex)
27. **M6**: "M6: Multi-Modality-to-Multi-Modality Multitask Mega-transformer for Unified Pretraining". `KDD(2021)` [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3447548.3467206)
28. "Probing Inter-modality: Visual Parsing with Self-Attention for Vision-and-Language Pre-training". `NeurIPS(2021)` [[PDF]](https://proceedings.neurips.cc/paper/2021/file/23fa71cc32babb7b91130824466d25a5-Paper.pdf)
29. **GilBERT**: "GilBERT: Generative Vision-Language Pre-Training for Modality-Incomplete Visual-Linguistic Tasks". `SIGIR(2021)` [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3404835.3462838)

### Information Retrieval

1. **ORQA**: "Latent Retrieval for Weakly Supervised Open Domain Question Answering". `ACL(2019)` [[PDF]](https://aclanthology.org/P19-1612.pdf)
2. **REALM**: "REALM: Retrieval-Augmented Language Model Pre-Training". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2002.08909)
3. **RAG**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks". `NeurIPS(2020)` [[PDF]](https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) [[Code]](https://github.com/huggingface/transformers/blob/master/examples/rag/)
4. **DPR**: "Dense Passage Retrieval for Open-Domain Question Answering". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.emnlp-main.550.pdf) [[Code]](https://github.com/facebookresearch/DPR)
5. "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering". `EACL(2021)` [[PDF]](https://aclanthology.org/2021.eacl-main.74.pdf) [[Code]](https://github.com/facebookresearch/FiD)

### Code

1. **CodeT5**: "CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.685.pdf) [[Code]](https://github.com/salesforce/CodeT5)
2. **Codex**: "Evaluating Large Language Models Trained on Code". `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2107.03374) [[Code]](https://www.github.com/openai/human-eval)

### Others

1. **ReasonBERT**: "ReasonBERT: Pre-trained to Reason with Distant Supervision". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.494.pdf) [[Code]](https://github.com/sunlab-osu/ReasonBERT)
2. "Sentence Bottleneck Autoencoders from Transformer Language Models". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.137.pdf) [[Code]](https://github.com/ivanmontero/autobot)
3. "Numeracy enhances the Literacy of Language Models". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.557.pdf) [[Code]](https://git.io/JuZXn)
4. **EnsLM**: "EnsLM: Ensemble Language Model for Data Diversity by Semantic Clustering". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.230.pdf) [[Code]](https://github.com/BoChenGroup/EnsLM)
5. "Reflective Decoding: Beyond Unidirectional Generation with Off-the-Shelf Language Models". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.114.pdf) [[Code]](https://homes.cs.washington.edu/~pawest/ReflectiveDecoding.html)
6. **BERTAC**: "BERTAC: Enhancing Transformer-based Language Models with Adversarially Pretrained Convolutional Neural Networks". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.164.pdf) [[Code]](https://github.com/nict-wisdom/bertac)
7. "Natural Language Understanding with Privacy-Preserving BERT". `CIKM(2021)` [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3459637.3482281)
8. **BANG**: "BANG: Bridging Autoregressive and Non-autoregressive Generation with Large Scale Pretraining". `ICML(2021)` [[PDF]](http://proceedings.mlr.press/v139/qi21a/qi21a.pdf) [[Code]](https://github.com/microsoft/BANG) 

## PLM Analysis

### Knowledge

1. "What Does BERT Look at? An Analysis of BERT’s Attention". `BlackBoxNLP(2019)` [[PDF]](https://aclanthology.org/W19-4828.pdf) [[Code]](https://github.com/clarkkev/attention-analysis)
2. "BERT Rediscovers the Classical NLP Pipeline". `ACL(2019)` [[PDF]](https://aclanthology.org/P19-1452.pdf)
3. "How Multilingual is Multilingual BERT?". `ACL(2019)` [[PDF]](https://aclanthology.org/P19-1493.pdf)
4. "A Structural Probe for Finding Syntax in Word Representations". `NAACL(2019)` [[PDF]](https://aclanthology.org/N19-1419.pdf) [[Code]](https://github.com/john-hewitt/structural-probes)
5. "Language Models as Knowledge Bases?". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1250.pdf) [[Code]](https://github.com/facebookresearch/LAMA)
6. "What Does BERT Learn about the Structure of Language?". `ACL(2019)` [[PDF]](https://aclanthology.org/P19-1356.pdf) [[Code]](https://github.com/ganeshjawahar/)
7. "Linguistic Knowledge and Transferability of Contextual Representations". `NAACL(2019)` [[PDF]](https://aclanthology.org/N19-1112.pdf)
8. "Assessing BERT's Syntactic Abilities". `arXiv(2019)` [[PDF]](https://arxiv.org/pdf/1901.05287.pdf) [[Code]](https://github.com/yoavg/bert-syntax)
9. "Probing Neural Network Comprehension of Natural Language Arguments" `ACL(2019)` [[PDF]](https://aclanthology.org/P19-1459.pdf)
10. "How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1006.pdf)
11. "Visualizing and Measuring the Geometry of BERT". `NeurIPS(2019)` [[PDF]](https://proceedings.neurips.cc/paper/2019/file/159c1ffe5b61b41b3c4d8f4c2150f6c4-Paper.pdf)
12. "Designing and Interpreting Probes with Control Tasks". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1275.pdf)
13. "Open Sesame: Getting inside BERT’s Linguistic Knowledge". `BlackboxNLP(2019)` [[PDF]](https://aclanthology.org/W19-4825.pdf) [[Code]](https://github.com/yongjie-lin/bert-opensesame)
14. "What do you learn from context? Probing for sentence structure in contextualized word representations". `ICLR(2019)` [[PDF]](https://openreview.net/pdf?id=SJzSgnRcKX) [[Code]](https://github.com/jsalt18-sentence-repl/jiant)
15. "Commonsense Knowledge Mining from Pretrained Models". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1109.pdf)
16. "Do NLP Models Know Numbers? Probing Numeracy in Embeddings". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1534.pdf)
17. "On the Cross-lingual Transferability of Monolingual Representations". `ACL(2020)` [[PDF]](https://aclanthology.org/2020.acl-main.421.pdf)
18. "Cross-Lingual Ability of Multilingual BERT: An Empirical Study". `ICLR(2020)` [[PDF]](https://openreview.net/pdf?id=HJeT3yrtDr) [[Code]](https://github.com/CogComp/mbert-study)
19. "What BERT Is Not: Lessons from a New Suite of Psycholinguistic Diagnostics for Language Models". `TACL(2020)` [[PDF]](https://aclanthology.org/2020.tacl-1.3.pdf) [[Code]](https://github.com/aetting/lm-diagnostics)
20. "How Much Knowledge Can You Pack Into the Parameters of a Language Model?". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.emnlp-main.437.pdf) [[Code]](https://goo.gle/t5-cbqa)
21. "How Can We Know What Language Models Know?". `TACL(2020)` [[PDF]](https://aclanthology.org/2020.tacl-1.28.pdf) [[Code]](https://github.com/jzbjyb/LPAQA)
22. "oLMpics-On What Language Model Pre-training Captures". `TACL(2020)` [[PDF]](https://aclanthology.org/2020.tacl-1.48.pdf) [[Code]](http://github.com/alontalmor/oLMpics)
23. "Information-Theoretic Probing with Minimum Description Length". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.emnlp-main.14.pdf) [[Code]](https://github.com/)
24. "Inducing Relational Knowledge from BERT". `AAAI(2020)` [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/6242/6098)
25. **AutoPrompt**: "AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.emnlp-main.346.pdf) [[Code]](http://ucinlp.github.io/autoprompt)
26. "Emergent linguistic structure in artificial neural networks trained by self-supervision". `PNAS(2020)` [[PDF]](https://www.pnas.org/content/pnas/117/48/30046.full.pdf)
27. "Evaluating Commonsense in Pre-Trained Language Models". `AAAI(2020)` [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/6523/6379) [[Code]](https://github.com/XuhuiZhou/CATS)
28. "Inducing Relational Knowledge from BERT". `AAAI(2020)` [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/6242/6098)
29. "Editing Factual Knowledge in Language Models". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.522.pdf) [[Code]](https://github.com/nicola-decao/KnowledgeEditor)
30. "How much pretraining data do language models need to learn syntax?". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.118.pdf)
31. "Stepmothers are mean and academics are pretentious: What do pretrained language models learn about you?". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.111.pdf) [[Code]](https://github.com/RochelleChoenni/stereotypes_in_lms)
32. "Putting Words in BERT's Mouth: Navigating Contextualized Vector Spaces with Pseudowords". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.806.pdf) [[Code]](https://github.com/tai314159/PWIBM-Putting-Words-in-Bert-s-Mouth)
33. "Frequency Effects on Syntactic Rule Learning in Transformers". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.72.pdf) [[Code]](https://github.com/google-research/language/tree/master/language/bertology/frequency_effects)
34. "Exploring the Role of BERT Token Representations to Explain Sentence Probing Results". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.61.pdf) [[Code]](https://github.com/hmohebbi/explain-probing-results)
35. "How is BERT surprised? Layerwise detection of linguistic anomalies". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.325.pdf) [[Code]](https://github.com/SPOClab-ca/layerwise-anomaly)
36. "Implicit Representations of Meaning in Neural Language Model". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.143.pdf) [[Code]](https://github.com/belindal/state-probes)
37. "Knowledgeable or Educated Guess? Revisiting Language Models as Knowledge Bases". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.146.pdf) [[Code]](https://github.com/c-box/LANKA)

### Robustness

1. "Universal Adversarial Triggers for Attacking and Analyzing NLP". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1221.pdf) [[Code]](https://github.com/Eric-Wallace/universal-triggers)
2. "Pretrained Transformers Improve Out-of-Distribution Robustness". `ACL(2020)` [[PDF]](https://aclanthology.org/2020.acl-main.244.pdf) [[Code]](https://github.com/camelop/NLP-Robustness)
3. **BERT-ATTACK**: "BERT-ATTACK: Adversarial Attack Against BERT Using BERT". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.emnlp-main.500.pdf) [[Code]](https://github.com/LinyangLee/BERT-Attack)
4. "Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment". `AAAI(2020)` [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/6311/6167) [[Code]](https://github.com/jind11/TextFooler)
5. "The Devil is in the Detail: Simple Tricks Improve Systematic Generalization of Transformers". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.49.pdf) [[Code]](https://github.com/robertcsordas/transformer_generalization)
6. "Sorting through the noise: Testing robustness of information processing in pre-trained language models". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.119.pdf) [[Code]](https://github.com/lalchand-pandia/Sorting-Through-The-Noise)

### Sparsity

1. "Are Sixteen Heads Really Better than One?". `NeurIPS(2019)` [[PDF]](https://proceedings.neurips.cc/paper/2019/file/2c601ad9d2ff9bc8b282670cdd54f69f-Paper.pdf) [[Code]](https://github.com/pmichel31415/are-16-heads-really-better-than-1)
2. "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned". `ACL(2019)` [[PDF]](https://aclanthology.org/P19-1580.pdf) [[Code]](https://github.com/lena-voita/the-story-of-heads)
3. "Revealing the Dark Secrets of BERT". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1445.pdf)
4. "The Lottery Ticket Hypothesis for Pre-trained BERT Networks". `NeurIPS(2020)` [[PDF]](https://proceedings.neurips.cc/paper/2020/file/b6af2c9703f203a2794be03d443af2e3-Paper.pdf) [[Code]](https://github.com/VITA-Group/BERT-Tickets)
5. "When BERT Plays the Lottery, All Tickets Are Winning". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.emnlp-main.259.pdf) [[Code]](https://github.com/sai-prasanna/bert-experiments)

### Others

1. "Scaling Laws for Neural Language Models". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2001.08361.pdf)
2. "Extracting Training Data from Large Language Models". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2012.07805.pdf) [[Code]](https://github.com/ftramer/LM_Memorization)
3. "On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? 🦜". `FACCT(2021)` [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922)
4. "Extracting Training Data from Large Language Models". `USENIX(2021)` [[PDF]](https://www.usenix.org/system/files/sec21-carlini-extracting.pdf) [[Code]](https://github.com/ftramer/LM_Memorization)
5. "Masked Language Modeling and the Distributional Hypothesis: Order Word Matters Pre-training for Little". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.230.pdf) [[Code]](https://github.com/pytorch/fairseq/tree/master/examples/shuffled_word_order)
6. "Effects of Parameter Norm Growth During Transformer Training: Inductive Bias from Gradient Descent". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.133.pdf) [[Code]](https://github.com/viking-sudo-rm/norm-growth)
7. "Discretized Integrated Gradients for Explaining Language Models". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.805.pdf) [[Code]](https://github.com/INK-USC/DIG)
8. "Do Long-Range Language Models Actually Use Long-Range Context?". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.62.pdf)
9. "Surface Form Competition: Why the Highest Probability Answer Isn’t Always Right". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.564.pdf) [[Code]](https://github.com/peterwestuw/surface-form-competition)
10. "Incorporating Residual and Normalization Layers into Analysis of Masked Language Models". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.373.pdf) [[Code]](https://github.com/gorokoba560/norm-analysis-of-transformer)
11. "Sequence Length is a Domain: Length-based Overfitting in Transformer Models". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.650.pdf)
12. "Are Pretrained Convolutions Better than Pretrained Transformers?". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.335.pdf)
13. "Positional Artefacts Propagate Through Masked Language Model Embeddings". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.413.pdf)
14. "When Do You Need Billions of Words of Pretraining Data?". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.90.pdf) [[Code]](https://github.com/nyu-mll/pretraining-learning-curves)
15. "BERT is to NLP what AlexNet is to CV: Can Pre-Trained Language Models Identify Analogies?". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.280.pdf) [[Code]](https://github.com/asahi417/analogy-language-model)
16. "Examining the Inductive Bias of Neural Language Models with Artificial Languages". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.38.pdf) [[Code]](https://github.com/rycolab/artificial-languages)
17. "Why Do Pretrained Language Models Help in Downstream Tasks? An Analysis of Head and Prompt Tuning". `NeurIPS(2021)` [[PDF]](https://proceedings.neurips.cc/paper/2021/file/86b3e165b8154656a71ffe8a327ded7d-Paper.pdf) 

## Efficient PLM

### Training

1. **RoBERTa**: "RoBERTa: A Robustly Optimized BERT Pretraining Approach". `arXiv(2019)` [[PDF]](https://arxiv.org/pdf/1907.11692) [[Code]](https://github.com/pytorch/fairseq)
2. "Efficient Training of BERT by Progressively Stacking". `ICML(2019)` [[PDF]](http://proceedings.mlr.press/v97/gong19a/gong19a.pdf) [[Code]](https://github.com/gonglinyuan/StackingBERT)
3. **Megatron-LM**: "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism". `arXiv(2019)` [[PDF]](https://arxiv.org/pdf/1909.08053.pdf) [[Code]](https://github.com/NVIDIA/Megatron-LM)
4. **ELECTRA**: "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators". `ICLR(2020)` [[PDF]](https://openreview.net/pdf?id=r1xMH1BtvB) [[Code]](https://github.com/google-research/electra)
5. "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes". `ICLR(2020)` [[PDF]](https://openreview.net/pdf?id=Syx4wnEtvH) [[Code]](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/lamb.py)
6. **GShard**: "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2006.16668.pdf)
7. **Admin**: "Understanding the Difficulty of Training Transformers". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.emnlp-main.463.pdf) [[Code]](https://github.com/LiyuanLucasLiu/Transforemr-Clinic)
8. **ZeRO**: "ZeRO: Memory optimizations Toward Training Trillion Parameter Models". `SC20: International Conference for High Performance Computing, Networking, Storage and Analysis` [[PDF]](https://ieeexplore.ieee.org/abstract/document/9355301) [[Code]](https://github.com/microsoft/deepspeed)
9. **Switch Transformers**: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity". `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2101.03961) [[Code]](https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py)
10. "How to Train BERT with an Academic Budget". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.831.pdf)
11. "Optimizing Deeper Transformers on Small Datasets". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.163.pdf) [[Code]](https://github.com/BorealisAI/DT-Fixup)
12. "EarlyBERT: Efficient BERT Training via Early-bird Lottery Tickets". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.171.pdf) [[Code]](https://github.com/VITA-Group/EarlyBERT)

### Inference

1. "BERT Loses Patience: Fast and Robust Inference with Early Exit". `NeurIPS(2020)` [[PDF]](https://proceedings.neurips.cc/paper/2020/file/d4dd111a4fd973394238aca5c05bebe3-Paper.pdf) [[Code]](https://github.com/JetRunner/PABEE)
2. **GAML-BERT**: "GAML-BERT: Improving BERT Early Exiting by Gradient Aligned Mutual Learning". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.242.pdf)
3. "Efficient Nearest Neighbor Language Models". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.461.pdf) [[Code]](https://github.com/jxhe/efficient-knnlm)
4. **GhostBERT**: "GhostBERT: Generate More Features with Cheap Operations for BERT". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.509.pdf) [[Code]](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/nlp/ghostbert)
5. **LeeBERT**: "LeeBERT: Learned Early Exit for BERT with cross-level optimization". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.231.pdf)
6. "Length-Adaptive Transformer: Train Once with Length Drop, Use Anytime with Search". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.508.pdf) [[Code]](https://github.com/clovaai/length-adaptive-transformer)
7. "Distilling Knowledge from BERT into Simple Fully Connected Neural Networks for Efficient Vertical Retrieval". `CIKM(2021)` [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3459637.3481909)

### Compression

1. **DistilBERT**: "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter". `arXiv(2019)` [[PDF]](https://arxiv.org/pdf/1910.01108) [[Code]](https://github.com/huggingface/transformers)
2. **PKD**: "Patient Knowledge Distillation for BERT Model Compression". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1441.pdf) [[Code]](https://github.com/intersun/PKD-for-BERT-Model-Compression)
3. "Distilling Task-Specific Knowledge from BERT into Simple Neural Networks". `arXiv(2019)` [[PDF]](https://arxiv.org/pdf/1903.12136.pdf)
4. **Q8BERT**: "Q8BERT: Quantized 8Bit BERT". `5th Workshop on Energy Efficient Machine Learning and Cognitive Computing - NeurIPS 2019` [[PDF]](https://arxiv.org/pdf/1910.06188.pdf)
5. **ALBERT**: "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations". `ICLR(2020)` [[PDF]](https://openreview.net/pdf?id=H1eA7AEtvS) [[Code]](https://github.com/google-research/ALBERT)
6. **TinyBERT**: "TinyBERT: Distilling BERT for Natural Language Understanding". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.findings-emnlp.372.pdf) [[Code]](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT)
7. **Layerdrop**: "Reducing Transformer Depth on Demand with Structured Dropout". `ICLR(2020)` [[PDF]](https://openreview.net/pdf?id=SylO2yStDr) [[Code]](https://github.com/pytorch/fairseq/tree/master/examples/layerdrop)
8. **Q-BERT**: "Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT". `AAAI(2020)` [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/6409/6265)
9. **MobileBERT**: "MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices". `ACL(2020)` [[PDF]](https://aclanthology.org/2020.acl-main.195.pdf) [[Code]](https://github.com/google-research/google-research/tree/master/mobilebert)
10. "Compressing BERT: Studying the Effects of Weight Pruning on Transfer Learning". `5th Workshop on Representation Learning for NLP(2020)` [[PDF]](https://aclanthology.org/2020.repl4nlp-1.18.pdf) [[Code]](https://github.com/mitchellgordon95/bert-prune)
11. **MiniLM**: "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2002.10957.pdf) [[Code]](https://github.com/microsoft/unilm/tree/master/minilm)
12. **FastBERT**: "FastBERT: a Self-distilling BERT with Adaptive Inference Time". `ACL(2020)` [[PDF]](https://aclanthology.org/2020.acl-main.537.pdf) [[Code]](https://github.com/autoliuweijie/FastBERT)
13. **DeeBERT**: "DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference". `ACL(2020)` [[PDF]](https://aclanthology.org/2020.acl-main.204.pdf) [[Code]](https://github.com/castorini/DeeBERT)
14. "Compressing Large-Scale Transformer-Based Models: A Case Study on BERT". `TACL(2021)` [[PDF]](https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl_a_00413/1964006/tacl_a_00413.pdf)
15. "Winning the Lottery with Continuous Sparsification". `NeurIPS(2020)` [[PDF]](https://proceedings.neurips.cc/paper/2020/file/83004190b1793d7aa15f8d0d49a13eba-Paper.pdf) [[Code]](https://github.com/lolemacs/continuous-sparsification)
16. **SqueezeBERT**: "SqueezeBERT: What can computer vision teach NLP about efficient neural networks?". `SustaiNLP(2020)` [[PDF]](https://aclanthology.org/2020.sustainlp-1.17.pdf)
17. **Audio ALBERT**: "Audio Albert: A Lite Bert for Self-Supervised Learning of Audio Representation". `SLT(2021)` [[PDF]](https://ieeexplore.ieee.org/abstract/document/9383575) [[Code]](https://github.com/pohanchi/AALBERT)
18. **T2R**: "Finetuning Pretrained Transformers into RNNs". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.830.pdf) [[Code]](https://github.com/jungokasai/T2R/)
19. "Beyond Preserved Accuracy: Evaluating Loyalty and Robustness of BERT Compression". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.832/) [[Code]](https://github.com/JetRunner/beyond-preserved-accuracy)
20. **Meta-KD**: "Meta-KD: A Meta Knowledge Distillation Framework for Language Model Compression across Domains". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.236.pdf) [[Code]](https://github.com/alibaba/EasyTransfer/tree/master/scripts/metaKD)
21. "Super Tickets in Pre-Trained Language Models: From Model Compression to Improving Generalization". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.510.pdf) [[Code]](https://github.com/cliang1453/super-structured-lottery-tickets)
22. **BinaryBERT**: "BinaryBERT: Pushing the Limit of BERT Quantization". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.334.pdf) [[Code]](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/BinaryBERT)
23. **AutoTinyBERT**: "AutoTinyBERT: Automatic Hyper-parameter Optimization for Efficient Pre-trained Language Models". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.400.pdf) [[Code]](https://github.com/huawei-noah/Pretrained-Language-Model)
24. "Marginal Utility Diminishes: Exploring the Minimum Knowledge for BERT Knowledge Distillation". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.228.pdf) [[Code]](https://github.com/llyx97/Marginal-Utility-Diminishes)
25. "Enabling Lightweight Fine-tuning for Pre-trained Language Model Compression based on Matrix Product Operators". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.418.pdf) [[Code]](https://github.com/RUCAIBox/MPOP)
26. **NAS-BERT**: "NAS-BERT: Task-Agnostic and Adaptive-Size BERT Compression with Neural Architecture Search". `KDD(2021)` [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3447548.3467262)

## PLM Adaptation

### Two-Stage

1. "Sentence Encoders on STILTs: Supplementary Training on Intermediate Labeled-data Tasks". `arXiv(2018)` [[PDF]](https://arxiv.org/pdf/1811.01088.pdf) [[Code]](https://github.com/zphang/pytorch-pretrained-BERT)
2. "How to Fine-Tune BERT for Text Classification?". `CCL(2019)` [[PDF]](http://cips-cl.org/static/anthology/CCL-2019/CCL-19-141.pdf)
3. "Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks". `ACL(2020)` [[PDF]](https://aclanthology.org/2020.acl-main.740.pdf) [[Code]](https://github.com/allenai/dont-stop-pretraining)
4. "Intermediate-Task Transfer Learning with Pretrained Language Models: When and Why Does It Work?". `ACL(2020)` [[PDF]](https://aclanthology.org/2020.acl-main.467.pdf)
5. "What to Pre-Train on? Efficient Intermediate Task Selection". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.827.pdf) [[Code]](https://github.com/Adapter-Hub/efficient-task-transfer)
6. "On the Influence of Masking Policies in Intermediate Pre-training". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.573.pdf)
7. **TADPOLE**: "TADPOLE: Task ADapted Pre-Training via AnOmaLy DEtection". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.463.pdf)

### Multi-Task

1. **MT-DNN**: "Multi-Task Deep Neural Networks for Natural Language Understanding". `ACL(2019)` [[PDF]](https://aclanthology.org/P19-1441.pdf) [[Code]](https://github.com/namisan/mt-dnn)
2. "BAM! Born-Again Multi-Task Networks for Natural Language Understanding". `ACL(2019)` [[PDF]](https://aclanthology.org/P19-1595.pdf) [[Code]](https://github.com/google-research/google-research/tree/master/bam)
3. "Improving Multi-Task Deep Neural Networks via Knowledge Distillation for Natural Language Understanding". `arXiv(2019)` [[PDF]](https://arxiv.org/pdf/1904.09482.pdf) [[Code]](https://github.com/namisan/mt-dnn)
4. **GradTS**: "GradTS: A Gradient-Based Automatic Auxiliary Task Selection Method Based on Transformer Networks". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.455.pdf)
5. "What's in Your Head? Emergent Behaviour in Multi-Task Transformer Models". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.646.pdf)
6. **MTAdam**: "MTAdam: Automatic Balancing of Multiple Training Loss Terms". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.837.pdf)
7. **Muppet**: "Muppet: Massive Multi-task Representations with Pre-Finetuning". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.468.pdf)
8. "The Stem Cell Hypothesis: Dilemma behind Multi-Task Learning with Transformer Encoders". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.451.pdf) [[Code]](https://github.com/emorynlp/stem-cell-hypothesis)
9. **BERTGen**: "BERTGen: Multi-task Generation through BERT". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.503.pdf) [[Code]](https://github.com/ImperialNLP/BertGen)
10. "Parameter-efficient Multi-task Fine-tuning for Transformers via Shared Hypernetworks". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.47.pdf) [[Code]](https://github.com/rabeehk/hyperformer)

### Adapater

1. "BERT and PALs: Projected Attention Layers for Efficient Adaptation in Multi-Task Learning". `ICML(2019)` [[PDF]](http://proceedings.mlr.press/v97/stickland19a/stickland19a.pdf) [[Code]](https://github.com/AsaCooperStickland/Bert-n-Pals)
2. **Adapter**: "Parameter-Efficient Transfer Learning for NLP". `ICML(2019)` [[PDF]](http://proceedings.mlr.press/v97/houlsby19a/houlsby19a.pdf) [[Code]](https://github.com/google-research/adapter-bert)
3. **AdapterDrop**: "AdapterDrop: On the Efficiency of Adapters in Transformers". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.626.pdf)
4. "On the Effectiveness of Adapter-based Tuning for Pretrained Language Model Adaptation". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.172.pdf)
5. "Learning to Generate Task-Specific Adapters from Task Description". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-short.82.pdf) [[Code]](https://github.com/INK-USC/hypter)

### Prompt

1. **PET**: "Exploiting Cloze-Questions for Few-Shot Text Classification and Natural Language Inference". `EACL(2021)` [[PDF]](https://aclanthology.org/2021.eacl-main.20.pdf) [[Code]](https://github.com/timoschick/pet)
2. "It’s Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners". `NAACL(2021)` [[PDF]](https://aclanthology.org/2021.naacl-main.185.pdf) [[Code]](https://github.com/timoschick/pet)
3. "Prefix-Tuning: Optimizing Continuous Prompts for Generation". `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2101.00190.pdf)
4. **LM-BFF**: "Making Pre-trained Language Models Better Few-shot Learners". `ACL(2021)` [[PDF]](https://arxiv.org/pdf/2012.15723) [[Code]](https://github.com/princeton-nlp/LM-BFF)
5. "What Makes Good In-Context Examples for GPT-3?". `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2101.06804.pdf) [[Code]](https://github.com/google-research/language/tree/master/language/totto)
6. "The Power of Scale for Parameter-Efficient Prompt Tuning". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.243.pdf) [[Code]](https://github.com/google-research/prompt-tuning)
7. "Finetuned Language Models Are Zero-Shot Learners". `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2109.01652)
8. "Calibrate Before Use: Improving Few-shot Performance of Language Models". `ICML(2021)` [[PDF]](http://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf) [[Code]](https://www.github.com/tonyzhaozh/few-shot-learning)
9. **TransPrompt**: "TransPrompt: Towards an Automatic Transferable Prompting Framework for Few-shot Text Classification". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.221.pdf) [[Code]](https://github.com/alibaba/EasyTransfer)
10. **SFLM**: "Revisiting Self-training for Few-shot Learning of Language Model". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.718.pdf) [[Code]](https://github.com/MatthewCYM/SFLM)
11. **ADAPET**: "Improving and Simplifying Pattern Exploiting Training". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.407.pdf) [[Code]](https://github.com/rrmenon10/ADAPET)

### Others

1. "To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks". `RepL4NLP(2019)` [[PDF]](https://aclanthology.org/W19-4302.pdf)
2. "An Embarrassingly Simple Approach for Transfer Learning from Pretrained Language Models". `NAACL(2019)` [[PDF]](https://aclanthology.org/N19-1213.pdf) [[Code]](https://github.com/alexandra-chron/siatl)
3. "Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2002.06305.pdf)
4. **SMART**: "SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.acl-main.197.pdf) [[Code]](https://github.com/namisan/mt-dnn)
5. "Revisiting Few-sample BERT Fine-tuning". `ICLR(2021)` [[PDF]](https://openreview.net/pdf?id=cO1IH43yUF)
6. **Mirror-BERT**: "Fast, Effective, and Self-Supervised: Transforming Masked Language Models into Universal Lexical and Sentence Encoders". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.109.pdf) [[Code]](https://github.com/cambridgeltl/mirror-bert)
7. "Pre-train or Annotate? Domain Adaptation with a Constrained Budget". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.409.pdf) [[Code]](https://github.com/bflashcp3f/ProcBERT)
8. **AVocaDo**: "AVocaDo: Strategy for Adapting Vocabulary to Downstream Domain". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.385.pdf)
9. **CHILD-TUNING**: "Raise a Child in Large Language Model: Towards Effective and Generalizable Fine-tuning". `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.749.pdf) [[Code]](https://github.com/alibaba/AliceMind/tree/main/ChildTuning)
10. "Taming Pre-trained Language Models with N-gram Representations for Low-Resource Domain Adaptation". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.259.pdf) [[Code]](https://github.com/shizhediao/T-DNA)
11. **LexFit**: "LexFit: Lexical Fine-Tuning of Pretrained Language Models". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.410.pdf) [[Code]](https://github.com/cambridgeltl/lexfit)
12. "Selecting Informative Contexts Improves Language Model Fine-tuning". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.87.pdf) [[Code]](https://github.com/HuthLab/IGF)
13. "An Empirical Study on Hyperparameter Optimization for Fine-Tuning Pre-trained Language Models". `ACL(2021)` [[PDF]](https://aclanthology.org/2021.acl-long.178.pdf) [[Code]](https://github.com/microsoft/FLAML/tree/main/flaml/nlp/)
14. "How Should Pre-Trained Language Models Be Fine-Tuned Towards Adversarial Robustness?". `NeurIPS(2021)` [[PDF]](https://proceedings.neurips.cc/paper/2021/file/22b1f2e0983160db6f7bb9f62f4dbb39-Paper.pdf) [[Code]](https://github.com/dongxinshuai/RIFT-NeurIPS2021)