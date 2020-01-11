# S-NET
  * A Tensorflow implementation of [S-NET: FROM ANSWER EXTRACTION TO ANSWER GENERATION FOR MACHINE READING COMPREHENSION](https://arxiv.org/pdf/1706.04815.pdf). This project is specially designed for the [MS-MARCO](https://arxiv.org/pdf/1611.09268.pdf) dataset.

## Requirements

#### General
  * Python >= 3.4
  * unzip, wget
#### Python Packages
  * tensorflow-gpu == 1.4.0
  * spaCy >= 2.0.0
  * tqdm
  * ujson

## HOW TO RUN

* Download the Question Answering (V1.1) [dataset](http://www.msmarco.org/dataset.aspx).
* Extract in `~/data/msmarco`
* Download [GloVe 840B 300d](http://nlp.stanford.edu/data/glove.840B.300d.zip)
* Extract in `~/data/glove`

To preprocess the data, run

```bash
# preprocess the data
python config_msm.py --mode analyze
```

Hyper parameters are stored in config_msm.py. To train/test the model, run

```bash
python config_msm.py --mode train/test
```

# S-NET IMPLEMENTATION TRACKER:
(Best model version is linked) 

### Evidence Extraction
---
1. :heavy_exclamation_mark: snet_without_pr: SNET without passage ranking
	* Hard coding tensorflow build using Pythonic for-loops on preconfigured number of paragraphs, regardless of how many actual paragraphs are there. **Inefficient.**
2. :heavy_exclamation_mark: snet_with_pr: SNET with passage ranking
	* Same as 1, but with passage ranking.
3. :heavy_exclamation_mark: snet_without_pr_test: SNET without passage ranking ???
4. :heavy_exclamation_mark: snet_without_pr_para_strip: (model2.py)
	* Complex method, which dynamically strips the padding by using tf while loops on the predicted y1 (start) and y2 (end) variable.
	* **Bad idea** since it messes with the start and end positions.
5. :heavy_exclamation_mark: snet_without_pr_para_strip_vp: (model2.py)
	* Dynamically strips the padding using a mask by using tf while loops on the predicted vps.
	* Somewhat better idea to strip paddings of vps using a precomputed mask and then continue to the rest of the model. 
6. :+1: snet_refactored_from_rnet: (model2.py)
	* Tried a different approach from scratch rnet code, much cleaner implementation which also solved infinite loss issue during training. What I did was used python for loop to build the model until the maximum allowed paragraphs as saved in the config.
	* I had to use Python for loop for that, since it needs to run the tf.cond at least once, I needed instantiated RNN objects. But also, inside the code, if the actual para count is less then the max allowed limit, then when actual para number gets exceeded during tensorflow training session, vp is not computed for those indexes and previous index's results are returned (DUMMY function).
	* After that, vps are again sliced using a boolean mask for representing all paras as a single concatenated one without any padding in between.

7. :+1: [snet_refactored_from_rnet_with_pr](https://github.com/burglarhobbit/machine-reading-comprehension/tree/master/S-NET/6_snet_refactored_from_rnet_with_pr) (model2.py)
	* Same as above, but with passage ranking
8. :large_blue_circle: snet_refactored_from_rnet_with_pr_floyd -> code for running on floyd cloud
9. :large_blue_circle: snet_refactored_from_rnet_with_pr_newgru -> ???

### Answer Synthesis
---

10. :large_blue_circle: nmt
11. :large_blue_circle: nmt_snet
12. :large_blue_circle: nmt_snet_2
	* working with question data integration. (No model changes)
13. :large_blue_circle: nmt_snet_3
	* working with question data integration. (With all model changes)
	* Pending: append fs and fe vectors.
14. :large_blue_circle: nmt_snet_4
	* working with question data integration + fs,fe vector append. (With all model changes)
15. :+1: [nmt_snet_5](https://github.com/burglarhobbit/machine-reading-comprehension/tree/master/S-NET/nmt_snet_5)
	* working with snet_ee (6, 7) data integration
16. :heavy_exclamation_mark: nmt_snet_ans_syn (idk?)
17. :heavy_exclamation_mark: snet_ee
	* old, irrelevant
18. :heavy_exclamation_mark: snet_ee2
	* old, irrelevant
19. :heavy_exclamation_mark: snet_ee3
	* old, irrelevant
20. :heavy_exclamation_mark: snet_pr_multipara
	* old, irrelevant
21. :heavy_exclamation_mark: snet_with_answer_synthesis
	* old, irrelevant

* Acknowledgements: The implementation of Evidence Extraction is derived from [R-NET on SQuAD](https://github.com/HKUST-KnowComp/R-Net). 
* The implementation of Answer Synthesis is derived from: [Neural Machine Translation](https://github.com/tensorflow/nmt)