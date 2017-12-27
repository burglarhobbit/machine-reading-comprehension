# machine-reading-comprehension

## S-NET

- A Tensorflow implementation of [S-NET: FROM ANSWER EXTRACTION TO ANSWER GENERATION FOR MACHINE READING COMPREHENSION](https://arxiv.org/pdf/1706.04815.pdf)
- This implementation is specifically designed for [MS-MARCO](http://www.msmarco.org/), a large-scale dataset drawing attention in the field of QA recently.

## Usage

1. First we need to download [MS-MARCO](http://www.msmarco.org/dataset.aspx) as well as the pre-trained [GloVe](nlp.stanford.edu/projects/glove/) embeddings

2. Data preprocessing, including tokenizing and collection of pre-trained word embeddings.
Two kinds of files, `{data/shared}_{train/dev}.json`, will be generated and stored in `Data`

~~~~
$ python3
>> from preprocess import generate_seq
>> generate_seq('train')
~~~~

3. Train S-NET evidence extraction by simply executing the following.

~~~~
python3 snet.py
~~~~

4. To be added after more results and analysis...