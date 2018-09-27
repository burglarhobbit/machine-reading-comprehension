1_snet_without_pr -> SNET without passage ranking
	Hard coding tensorflow build using Pythonic for-loops on preconfigured number of paragraphs, regardless of how many 
	actual paragraphs are there.
2_snet_with_pr -> SNET with passage ranking
	Same as 1, but with passage ranking.
3_snet_without_pr_test -> SNET without passage ranking ???
4_snet_without_pr_para_strip -> (model2.py) complex method, which dynamically strips the padding by using tf while loops
	on the predicted y1 (start) and y2 (end) variable. Bad idea since it messes with the start and end positions.
4_snet_without_pr_para_strip_vp -> (model2.py) dynamically strips the padding using a mask by using tf while loops
	on the predicted vps. Somewhat better idea to strip paddings of vps using a precomputed mask and then continue to the rest of the model. 
5_snet_refactored_from_rnet -> (model2.py) -> tried a different approach from scratch rnet code, much cleaner implementation which also solved infinite loss issue during training. What I did was used python for loop to build the model until the maximum allowed paragraphs as saved in the config. I had to use Python for loop for that, since it needs to run the tf.cond at least once, I needed instantiated RNN objects. But also, inside the code, if the actual para count is less then the max allowed limit, then when actual para number gets exceeded during tensorflow training session, vp is not computed for those indexes and previous index's results are returned (DUMMY function). After that, vps are again sliced using a boolean mask for representing all paras as a single concatenated one without any padding in between.
6_snet_refactored_from_rnet_with_pr -> same as above, but with passage ranking
6_snet_refactored_from_rnet_with_pr_floyd -> code for running on floyd cloud
6_snet_refactored_from_rnet_with_pr_newgru -> 
nmt
nmt_snet
nmt_snet_2 -> working with question data integration. (No model changes)
nmt_snet_3 ->  working with question data integration. (With all model changes) Pending: append fs and fe vectors.
nmt_snet_4 -> working with question data integration + fs,fe vector append. (With all model changes)
nmt_snet_5 -> working with snet_ee data integration
nmt_snet_ans_syn
snet_ee -> old, irrelevant
snet_ee2 -> -> old, irrelevant
snet_ee3 -> -> old, irrelevant
snet_pr_multipara -> -> old, irrelevant
snet_with_answer_synthesis -> -> old, irrelevant