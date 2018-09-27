global_span -> rouge-l w.r.t 8 paragraphs. Loss shoots to infinite values.
local_span -> rouge-l w.r.t only 1 paragraphs. Loss is converging in all local span models.
local_span_with_high_dim -> increased the dimensions of hidden units -> no significant changes
local_span_with_new_initialization_values -> new type of initialization -> degraded performance
local_span_with_var_summary -> variable summaries
local_span_with_var_summary_tf_apis -> variable summaries + saving attention logits for visualization (model.py)
no-outlier -> old model, global_span type, irrelevantanalyze_dataset.py
a.out
config_msm.py
download.sh
evaluate-v1.1.py
func.py
global_span
local_span
local_span_single_para
local_span_with_high_dim
local_span_with_new_initialization_values
local_span_with_var_summary
local_span_with_var_summary_tf_apis
main.py
model.py
outputs.txt
prepro_msm.py
__pycache__
README.md
rouge_score.py
run_file.sh
util.py
