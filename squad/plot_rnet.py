import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sys import argv
import ujson as json
import spacy 
from tqdm import tqdm

from matplotlib import rc
rc('text', usetex=True)

nlp = spacy.blank("en")

def word_tokenize(sent):
	doc = nlp(sent)
	return [token.text for token in doc]

def convert_idx(text, tokens):
	current = 0
	spans = []
	for token in tokens:
		current = text.find(token, current)
		if current < 0:
			print("Token {} cannot be found".format(token))
			raise Exception()
		spans.append((current, current + len(token)))
		current += len(token)
	return spans


def process_squad(filename):
	examples = {}
	eval_examples = {}
	total = 0
	with open(filename, "r") as fh:
		source = json.load(fh)
		for article in tqdm(source["data"]):
			for para in article["paragraphs"]:
				context = para["context"].replace(
					"''", '" ').replace("``", '" ')
				context_tokens = word_tokenize(context)
				context_chars = [list(token) for token in context_tokens]
				spans = convert_idx(context, context_tokens)
				for qa in para["qas"]:
					total += 1
					ques = qa["question"].replace(
						"''", '" ').replace("``", '" ')
					ques_tokens = word_tokenize(ques)
					ques_chars = [list(token) for token in ques_tokens]
					y1s, y2s = [], []
					answer_texts = []
					for answer in qa["answers"]:
						answer_text = answer["text"]
						answer_start = answer['answer_start']
						answer_end = answer_start + len(answer_text)
						answer_texts.append(answer_text)
						answer_span = []
						for idx, span in enumerate(spans):
							if not (answer_end <= span[0] or answer_start >= span[1]):
								answer_span.append(idx)
						y1, y2 = answer_span[0], answer_span[-1]
						y1s.append(y1)
						y2s.append(y2)
					example = {"context_tokens": context_tokens, "context_chars": context_chars,\
						"ques_tokens": ques_tokens, "y1s": y1s[-1], "y2s": y2s[-1], \
						"id": total, "uuid": qa["id"], "answers": answer_texts, "ques": ques}
					examples[total] = example
		print("{} questions in total".format(len(examples)))
	return examples

if len(argv)>1:
	if argv[1] == 'squad':
		variables = np.load('variables_squad.npz')
else:
	variables = np.load('variables.npz')
start = variables['start']
end = variables['end']
match_logits = variables['match_logits']
#match_outputs = variables['match_outputs']
att_logits = variables['att_logits']
#att_outputs = variables['att_outputs']
qa_id = variables['qa_id']
batch_size = bs = 16
examples_per_batch = epb = 4

#examples = process_squad("dev-v1.1.json")
examples = json.loads(open("dev_eval_squad.json").read())

print(qa_id.shape)
print(match_logits.shape)
print(att_logits.shape)
#print(list(examples.keys()))
for i in range(0,len(qa_id)):
	#plt.imshow(match_logits[0], cmap='hot', interpolation='nearest')
	#plt.show()
	extra_zeros_rows_attention = np.trim_zeros(att_logits[i,:,0],trim='b').reshape((-1,1))
	extra_zeros_columns_attention = np.trim_zeros(att_logits[i,0,:],trim='b').reshape((-1,1))
	ezra = extra_zeros_rows_attention.shape[0]
	ezca = extra_zeros_columns_attention.shape[0] # limits padded question 

	extra_zeros_rows_match = np.trim_zeros(match_logits[i,:,0],trim='b').reshape((-1,1))
	extra_zeros_columns_match = np.trim_zeros(match_logits[i,0,:],trim='b').reshape((-1,1))
	ezrm = extra_zeros_rows_match.shape[0] 
	ezcm = extra_zeros_columns_match.shape[0] # limits padded para

	#print(att_logits[i,:,0].reshape((-1,1)).shape)
	#print(extra_zeros_rows_attention.shape)
	#print(ezra,ezca)
	#print(ezrm,ezcm)
	#print(extra_zeros_rows_attention)
	#match_logits_trimmed = match_logits[i,]
	print("Example:",i,"\n\n")
	#print(list(examples.keys()))
	qid = str(qa_id[i])
	if str(qa_id[i]) not in examples.keys():
		continue
	context_tokens = word_tokenize(examples[qid]["context"])
	ques_tokens = word_tokenize(examples[qid]["question"])

	labels = [r"$\textbf{" + "Gated Attention-based RNN Gradients" + "}$",
			  r"$\textbf{" + "Self-Matching Attention Gradients" + "}$",
			  r"$\textbf{" + "Answer Start and End position logits" + "}$" ]
	passage_label = "Passage Token"
	question_label = "Question Token"
	logits_label = "Start/End Probability"
	print("Passage:",i,"\n")
	for j,k in enumerate(context_tokens):
		print(j,":",k)
	print("Question:",ques_tokens,"\n")
	ans_start = np.argmax(start[i])
	ans_end = np.argmax(end[i])
	print("Start:",ans_start,"End:",ans_end)
	fig = plt.figure(1)
	ax1 = plt.subplot(311)
	##ax1 = sns.heatmap(att_logits[i,:ezra,:ezca].T) # higher unwanted info
	ax1 = sns.heatmap(att_logits[i,:ezcm,:ezca].T)
	ax1.set_title(labels[0])
	ax1.set_xlabel(passage_label)
	ax1.set_ylabel(question_label)
	
	plt.subplot(312)	
	##ax2 = sns.heatmap(match_logits[i,:ezrm,:ezcm].T)
	ax2 = sns.heatmap(match_logits[i,:ezcm,:ezcm])
	ax2.set_title(labels[1])
	ax2.set_xlabel(passage_label)
	ax2.set_ylabel(passage_label)

	#plt.figure(2)
	#plt.figure(2)
	ax3 = plt.subplot(313)
	#plt.subplot(313)
	space = "                           "
	space += space + space
	ax3.set_title(space+labels[2])
	xaxis = int(ezcm*600/480*25.4/27)
	#xaxis = ezcm
	y1 = start[i,:xaxis]
	y2 = end[i,:xaxis]
	x = np.arange(0,xaxis)
	#ax3.margins(x=0)
	print(ezcm)
	ax3.plot(x,y1,'g--',x,y2,'r--')
	#ax3.set_xlabel(passage_label)
	#ax3.set_ylabel("Probability")

	#ax3.set_label_coords(0.4,0.4)
	#plt.subplots_adjust(left=0.3, bottom=0, right=0.5, top=1, wspace=0, hspace=0)
	plt.tight_layout()
	#plt.savefig('squad_outputs/fig_'+str(i)+".png", dpi=1000, bbox_inches='tight')
	plt.show()
#print(qa_id.to_list())
