# R-NET IMPLEMENTATION TRACKER:

1. global_span
	* Applied ROUGE-L w.r.t all paragraphs (start-end index spans across paragraphs during preprocessing). 
	* Observation: Loss shoots to infinite value. Unstable. **Does not train**.
2. local_span
	* ROUGE-L span limited to only 1 paragraph during preprocessing. Loss is converging upto some point. **Stable**.
3. local_span_single_para
	* Only trained on 1 paragraph from every question from which the answer belongs. Stable but **not relevant**.
4. local_span_with_high_dim
	* Increased the dimensions of hidden units to speed up convergence. **Stable but** no significant changes from 2.
5. local_span_with_new_initialization_values 
	* New type of randomized initializations of all layers. **Degraded performance**.
6. local_span_with_var_summary
	* Same as 2, but also saved variable summaries of most of the variables for tensorboard visualization.
7. local_span_with_var_summary_tf_apis -> variable summaries + saving attention logits for visualization (model.py)
	* Same as 6, but also saves attention outputs from 2 layers for later visualization.
8. no-outlier
	* An old model, global_span type, **irrelevant**.