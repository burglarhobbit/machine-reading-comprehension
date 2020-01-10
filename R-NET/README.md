# R-NET IMPLEMENTATION TRACKER:

:+1: 1. local_span
	* ROUGE-L span limited to only 1 paragraph during preprocessing. Loss is converging upto some point. **Stable**.

:heavy_exclamation_mark: 2. local_span_single_para
	* Only trained on 1 paragraph from every question from which the answer belongs. Stable but **not relevant**.
:heavy_exclamation_mark: 3. local_span_with_high_dim
	* Increased the dimensions of hidden units to speed up convergence. **Stable but** no significant changes from 2.
:-1: 4. local_span_with_new_initialization_values 
	* New type of randomized initializations of all layers. **Degraded performance**.
:large_blue_circle: 5. local_span_with_var_summary
	* Same as 2, but also saved variable summaries of most of the variables for tensorboard visualization.
:large_blue_circle: 6. local_span_with_var_summary_tf_apis -> variable summaries + saving attention logits for visualization (model.py)
	* Same as 6, but also saves attention outputs from 2 layers for later visualization.
---
7. global_span
	* Applied ROUGE-L w.r.t all paragraphs (start-end index spans across paragraphs during preprocessing). 
	* Observation: Loss shoots to infinite value. Unstable. **Does not train**.
[comment]: <> (8. no-outlier: An old model, global_span type, **irrelevant**.)