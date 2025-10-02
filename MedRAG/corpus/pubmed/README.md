---
task_categories:
- question-answering
language:
- en
tags:
- medical
- question answering
- large language model
- retrieval-augmented generation
size_categories:
- 10M<n<100M
---
# The PubMed Corpus in MedRAG

This HF dataset contains the snippets from the PubMed corpus used in [MedRAG](https://arxiv.org/abs/2402.13178). It can be used for medical Retrieval-Augmented Generation (RAG).

## News
- (02/26/2024) The "id" column has been reformatted. A new "PMID" column is added.

## Dataset Details

### Dataset Descriptions

[PubMed](https://pubmed.ncbi.nlm.nih.gov/) is the most widely used literature resource, containing over 36 million biomedical articles. 
For MedRAG, we use a PubMed subset of 23.9 million articles with valid titles and abstracts.
This HF dataset contains our ready-to-use snippets for the PubMed corpus, including 23,898,701 snippets with an average of 296 tokens.

### Dataset Structure
Each row is a snippet of PubMed, which includes the following features:

- id: a unique identifier of the snippet
- title: the title of the PubMed article from which the snippet is collected
- content: the abstract of the PubMed article from which the snippet is collected
- contents: a concatenation of 'title' and 'content', which will be used by the [BM25](https://github.com/castorini/pyserini) retriever

## Uses

<!-- Address questions around how the dataset is intended to be used. -->

### Direct Use

<!-- This section describes suitable use cases for the dataset. -->

```shell
git clone https://huggingface.co/datasets/MedRAG/pubmed
```

### Use in MedRAG

```python
>> from src.medrag import MedRAG

>> question = "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral"
>> options = {
    "A": "paralysis of the facial muscles.",
    "B": "paralysis of the facial muscles and loss of taste.",
    "C": "paralysis of the facial muscles, loss of taste and lacrimation.",
    "D": "paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation."
}

>> medrag = MedRAG(llm_name="OpenAI/gpt-3.5-turbo-16k", rag=True, retriever_name="MedCPT", corpus_name="PubMed")
>> answer, snippets, scores = medrag.answer(question=question, options=options, k=32) # scores are given by the retrieval system
```

## Citation
```shell
@article{xiong2024benchmarking,
    title={Benchmarking Retrieval-Augmented Generation for Medicine}, 
    author={Guangzhi Xiong and Qiao Jin and Zhiyong Lu and Aidong Zhang},
    journal={arXiv preprint arXiv:2402.13178},
    year={2024}
}
```