# R-NET on MS-MARCO Dataset
	* A Tensorflow implementation of [R-NET: MACHINE READING COMPREHENSION WITH SELF-MATCHING NETWORKS](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf). This project is specially designed for the [MS-MARCO](https://arxiv.org/pdf/1611.09268.pdf) dataset.

## Requirements

There have been a lot of known problems caused by using different software versions. Please check your versions before opening issues or emailing me.

#### General
  * Python >= 3.4
  * unzip, wget
#### Python Packages
  * tensorflow-gpu == 1.4.0
  * spaCy >= 2.0.0
  * tqdm
  * ujson

# HOW TO RUN

* Download the Question Answering (V1.1) [dataset](http://www.msmarco.org/dataset.aspx).
* Extract in `~/data/msmarco`
* Download[GloVe 840B 300d](http://nlp.stanford.edu/data/glove.840B.300d.zip)
* Extract in `~/data/glove`

To preprocess the data, run

```bash
# preprocess the data
python config_msm.py --mode analyze
```

Hyper parameters are stored in config.py. To debug/train/test the model, run

```bash
python config_msm.py --mode train/test
```

# R-NET IMPLEMENTATION TRACKER:

1. :+1: [local_span](https://github.com/burglarhobbit/machine-reading-comprehension/tree/master/R-NET/local_span)
	* ROUGE-L span limited to only 1 paragraph during preprocessing. Loss is converging upto some point. **Stable**.
2. :heavy_exclamation_mark: local_span_single_para
	* Only trained on 1 paragraph from every question from which the answer belongs. Stable but **not relevant**.
3. :heavy_exclamation_mark: local_span_with_high_dim
	* Increased the dimensions of hidden units to speed up convergence. **Stable but** no significant changes from 2.
4. :-1: local_span_with_new_initialization_values 
	* New type of randomized initializations of all layers. **Degraded performance**.
5. :large_blue_circle: local_span_with_var_summary
	* Same as 2, but also saved variable summaries of most of the variables for tensorboard visualization.
6. :large_blue_circle: local_span_with_var_summary_tf_apis -> variable summaries + saving attention logits for visualization (model.py)
	* Same as 6, but also saves attention outputs from 2 layers for later visualization.
---
7. :heavy_exclamation_mark: global_span
	* Applied ROUGE-L w.r.t all paragraphs (start-end index spans across paragraphs during preprocessing). 
	* Observation: Loss shoots to infinite value. Unstable. **Does not train**.
[comment]: <> (8. no-outlier: An old model, global_span type, **irrelevant**.)

* Acknowledgements: This implementation is derived from [R-NET on SQuAD](https://github.com/HKUST-KnowComp/R-Net).