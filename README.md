<p align="center">
    <img src="https://user-images.githubusercontent.com/7140210/72913702-d55a8480-3d3d-11ea-99fc-f2ef29af4e72.jpg" width="700"> </br>
    Marco Lodola, Monument to Umberto Eco, Alessandria 2019
</p>

# UmBERTo: an Italian Language Model trained with Whole Word Masking

UmBERTo is a Roberta-based Language Model trained on large Italian Corpora.
This implementation is based on Facebook Research AI code (https://github.com/pytorch/fairseq)

# Description

UmBERTo inherits from RoBERTa base model architecture which improves the initial BERT by identifying key hyperparameters for better results.
Umberto extends Roberta and uses two innovative approaches: ***SentencePiece*** and ***Whole Word Masking***.
SentencePiece Model (**SPM**) is a language-independent subword tokenizer and detokenizer designed for Neural-based text processing and creates sub-word units specifically to the size of the chosen vocabulary and the language of the corpus. 
Whole Word Masking (**WWM**) applies mask to an entire word, if at least one of all tokens created by SentencePiece Tokenizer was originally chosen as mask. So only entire word are masked, not subwords.

Two models are released:
  - **umberto-wikipedia-uncased-v1**, an uncased model trained on a relative small corpus (~7GB) extracted from 
  [Wikipedia-ITA](https://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/).
  - **umberto-commoncrawl-cased-v1**, a cased model trained on Commoncrawl ITA exploiting [OSCAR](https://traces1.inria.fr/oscar/) (Open Super-large Crawled ALMAnaCH coRpus) Italian large corpus ( ~69GB)

Both models have 12-layer, 768-hidden, 12-heads, 110M parameters (BASE).


| Model | WWM | CASED | TOKENIZER | VOCAB SIZE  | TRAIN STEPS | FAIRSEQ  | TRANSFORMERS |
| ------ | ------ | ------ | ------ | ------ |------ | ------ | --- |
| `umberto-wikipedia-uncased-v1` | YES  | NO | SPM | 32K | 100k | [Link](http://bit.ly/2s7JmXh)| [Link](http://bit.ly/35wbSj6) |
| `umberto-commoncrawl-cased-v1` | YES | YES | SPM | 32K | 125k | [Link](http://bit.ly/2TakHfJ)| [Link](http://bit.ly/35zO7GH) |

We trained both the models on 8 Nvidia V100 GPUs (p2.8xlarge P2 EC2 instance) during 4 days on [AWS Sagemaker](https://aws.amazon.com/it/sagemaker/).

# Installation

### Dependencies:
```
torch >= 1.3.1
sentencepiece
transformers
fairseq
```


#### Transformers


```pip install transformers```

To install transformers from original repo (TESTED):
```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .
```

#### Fairseq

To use a version of `fairseq` with UmBERTo support, build from source doing these steps:
```bash
git clone https://github.com/musixmatchresearch/fairseq
cd fairseq
pip install .
```


# Examples

### Transformers

From official [HuggingFace](https://github.com/huggingface/transformers) code.

#### with transformer pipeline

```python

from transformers import pipeline

fill_mask = pipeline(
	"fill-mask",
	model="Musixmatch/umberto-commoncrawl-cased-v1",
	tokenizer="Musixmatch/umberto-commoncrawl-cased-v1"
)

result = fill_mask("Umberto Eco è <mask> un grande scrittore")

#[{'sequence': '<s> Umberto Eco è considerato un grande scrittore</s>', 'score': 0.1859988570213318, 'token': 5032}, 
#{'sequence': '<s> Umberto Eco è stato un grande scrittore</s>', 'score': 0.1781671643257141, 'token': 471}, 
#{'sequence': '<s> Umberto Eco è sicuramente un grande scrittore</s>', 'score': 0.16565577685832977, 'token': 2654}, 
#{'sequence': '<s> UmbertoEco è indubbiamente un grande scrittore</s>', 'score': 0.09328985959291458, 'token': 17908}, 
#{'sequence': '<s> Umberto Eco è certamente un grande scrittore</s>', 'score': 0.05470150709152222, 'token': 5269}]
```

#### with transformer AutoTokenizer,AutoModel  
```python
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("Musixmatch/umberto-commoncrawl-cased-v1")
umberto = AutoModel.from_pretrained("Musixmatch/umberto-commoncrawl-cased-v1")

encoded_input = tokenizer.encode("Umberto Eco è stato un grande scrittore")
input_ids = torch.tensor(encoded_input).unsqueeze(0)  # Batch size 1
outputs = umberto(input_ids)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output
```

### Fairseq

```python
import torch

umberto = torch.hub.load('musixmatchresearch/umberto', 'umberto_commoncrawl_cased')

assert isinstance(umberto.model, torch.nn.Module)
umberto.eval()  # disable dropout (or leave in train mode to finetune)

# Masked LM Inference
masked_line = 'Umberto Eco è <mask> un grande scrittore'
result = umberto.fill_mask(masked_line, topk=20)
# Output:
#('Umberto Eco è considerato un grande scrittore', 0.19939924776554108, ' considerato'), 
#('Umberto Eco è sicuramente un grande scrittore', 0.1669664829969406, ' sicuramente'), 
#('Umberto Eco è stato un grande scrittore', 0.16225320100784302, ' stato'), 
#('Umberto Eco è indubbiamente un grande scrittore', 0.09528309106826782, ' indubbiamente')
...
```



# Results
We obtained state-of-the-art results for POS tagging, confirming that cased models trained with WWM perform better than uncased ones.
Our model `Umberto-Wikipedia-Uncased` trained with WWM on a smaller dataset and uncased, produces important results comparable to the cased results.

### Umberto-Wikipedia-Uncased
These results refers to umberto-wikipedia-uncased model.

#### Part of Speech (POS)

| Dataset | F1 | Precision | Recall | Accuracy |
| ------ | ------ | ------ |  ------ |  ------ |
| **UD_Italian-ISDT** | 98.563  | 98.508 | 98.618 | **98.717** | 
| **UD_Italian-ParTUT** | 97.810 | 97.835 |  97.784 | **98.060** | 

#### Named Entity Recognition (NER)

| Dataset | F1 | Precision | Recall | Accuracy |
| ------ | ------ | ------ |  ------ |  ----- |
| **ICAB-EvalITA07** | **86.240** | 85.939 | 86.544 | 98.534 | 
| **WikiNER-ITA** | **90.483** | 90.328 | 90.638 | 98.661 | 

### Umberto-Commoncrawl-Cased

These results refers to umberto-commoncrawl-cased model.

#### Part of Speech (POS)

| Dataset | F1 | Precision | Recall | Accuracy |
| ------ | ------ | ------ |  ------ |  ------ |
| **UD_Italian-ISDT** | 98.870  | 98.861 | 98.879 | **98.977** | 
| **UD_Italian-ParTUT** | 98.786 | 98.812 |  98.760 | **98.903** | 

#### Named Entity Recognition (NER)

| Dataset | F1 | Precision | Recall | Accuracy |
| ------ | ------ | ------ |  ------ |  ------ |
| **ICAB-EvalITA07** | **87.565**  | 86.596  | 88.556  | 98.690 | 
| **WikiNER-ITA** | **92.531**  | 92.509 | 92.553 | 99.136 | 



## References:
* BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding [Paper](https://arxiv.org/abs/1810.04805), [Github](https://github.com/google-research/bert)
* CamemBERT: a Tasty French Language Model [Paper](https://www.researchgate.net/publication/337183733_CamemBERT_a_Tasty_French_Language_Model), [Page](https://camembert-model.fr/)
* GilBERTo: An Italian pretrained language model based on RoBERTa [Github](https://github.com/idb-ita/GilBERTo)
* RoBERTa: A Robustly Optimized BERT Pretraining Approach [Paper](https://arxiv.org/abs/1907.11692), [Github](https://github.com/pytorch/fairseq/tree/master/fairseq/models)
* Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing [Paper](https://www.aclweb.org/anthology/D18-2012/), [Github](https://github.com/google/sentencepiece)
* Asynchronous Pipeline for Processing Huge Corpora on Medium to Low Resource Infrastructures [Paper](https://hal.inria.fr/hal-02148693), [Page]()
* Italy goes to Stanford: a collection of CoreNLP modules for Italian (TINT) [Paper](https://arxiv.org/abs/1609.06204), [Github](https://github.com/dhfbk/tint), [Page](https://dh.fbk.eu/technologies/tint-italian-nlp-tool) 


## Citation
All of the original datasets are publicly available or were released with the owners' grant. The datasets are all released under a CC0 or CCBY license.

* UD Italian-ISDT Dataset [Github](https://github.com/UniversalDependencies/UD_Italian-ISDT)
* UD Italian-ParTUT Dataset [Github](https://github.com/UniversalDependencies/UD_Italian-ParTUT)
* WIKINER [Page](https://figshare.com/articles/Learning_multilingual_named_entity_recognition_from_Wikipedia/5462500) , [Paper](https://www.sciencedirect.com/science/article/pii/S0004370212000276?via%3Dihub)
* I-CAB (Italian Content Annotation Bank), EvalITA [Page](http://www.evalita.it/)
```
@inproceedings {magnini2006annotazione,
	title = {Annotazione di contenuti concettuali in un corpus italiano: I - CAB},
	author = {Magnini,Bernardo and Cappelli,Amedeo and Pianta,Emanuele and Speranza,Manuela and Bartalesi Lenzi,V and Sprugnoli,Rachele and Romano,Lorenza and Girardi,Christian and Negri,Matteo},
	booktitle = {Proc.of SILFI 2006},
	year = {2006}
}
@inproceedings {magnini2006cab,
	title = {I - CAB: the Italian Content Annotation Bank.},
	author = {Magnini,Bernardo and Pianta,Emanuele and Girardi,Christian and Negri,Matteo and Romano,Lorenza and Speranza,Manuela and Lenzi,Valentina Bartalesi and Sprugnoli,Rachele},
	booktitle = {LREC},
	pages = {963--968},
	year = {2006},
	organization = {Citeseer}
}
```

## Acknowledgments
Special thanks to I-CAB (Italian Content Annotation Bank) and [EvalITA](http://www.evalita.it/) authors to provide the datasets as part of Master Thesis Research project with [School of Engineering, University of Bologna](https://www.unibo.it/en/university/campuses-and-structures/schools/school-of-engineering).

## Authors

**Loreto Parisi**: `loreto at musixmatch dot com`, [loretoparisi](https://github.com/loretoparisi)<br>
**Simone Francia**: `simone.francia at musixmatch dot com`, [simonefrancia](https://github.com/simonefrancia)<br>
**Paolo Magnani**: `paul.magnani95 at gmail dot com`, [paulthemagno](https://github.com/paulthemagno)<br>

## About Musixmatch AI
![Musxmatch Ai mac app icon-128](https://user-images.githubusercontent.com/163333/72244273-396aa380-35ee-11ea-894b-4ea48230c02b.png)<br>
We do Machine Learning and Artificial Intelligence @[musixmatch](https://twitter.com/Musixmatch)<br>
Follow us on [Twitter](https://twitter.com/musixmatchai) [Github](https://github.com/musixmatchresearch)


