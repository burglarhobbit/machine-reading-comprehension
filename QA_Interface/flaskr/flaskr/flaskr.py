import requests
import json
from flask import Flask, jsonify, redirect, url_for, request, render_template, Markup, flash
from flask import g
import sqlite3

#static_folder='/home/search/DrQA/scripts/pipeline/flaskr/flaskr/nvli-search'

app = Flask(__name__)

app.url_map.strict_slashes = False
app.add_url_rule('/nvli-search/<path:filename>', endpoint='nvli-search', view_func=app.send_static_file)

sample = """Top Predictions:
+------+-------------+-----------------------------+--------------+-----------+
| Rank |    Answer   |             Doc             | Answer Score | Doc Score |
+------+-------------+-----------------------------+--------------+-----------+
|  1   | 11 May 1938 | Narendra Patel, Baron Patel |  3.3312e+05  |   23.436  |
+------+-------------+-----------------------------+--------------+-----------+"""

PORT = '9001'
POST_URL = 'http://10.129.2.77:' + PORT + '/api'
MODEL_URL = 'http://10.129.2.77:' + PORT + '/get_models'
ALL_MODEL_URL = 'http://10.129.2.77:' + PORT + '/get_all_models'
RETRIEVAL_MODEL_URL = 'http://10.129.2.77:' + PORT + '/get_retrieval_models'
TABLE_HEADERS_URL = 'http://10.129.2.77:' + PORT + '/get_headers/'
FEEDBACK_URL = 'http://10.129.2.77:' + PORT + '/submit_feedback'
TIMEOUT = 60 # seconds
DATABASE = 'nvli.db'
error_string_timeout = '500 Internal Server Error: API temporarily unavailable (requests.exceptions.ReadTimeout)'
error_string_conn = '500 Internal Server Error: API temporarily unavailable (requests.exceptions.ConnectionError)'
BASE = '/nvli-search'

@app.errorhandler(requests.exceptions.ReadTimeout)
@app.errorhandler(requests.exceptions.ConnectionError)
def request_timeout(e):
	return jsonify(error=500,text=str(e)), 500

app.register_error_handler(requests.exceptions.ReadTimeout, lambda e: error_string_timeout)
app.register_error_handler(requests.exceptions.ConnectionError, lambda e: error_string_conn)


#@app.route('/success/',defaults={'name':'bhavya'})
@app.route('/results/<ques>/<topn>/<model>')
def results(ques, topn, model):

	headers = {'Content-Type': 'application/json'}
	ques_dict = jsonify(question = ques)
	ques_dict = json.dumps({'question':ques, 'topn': topn})
	answers = []
	
	if model == "drqa":	
		predictions = requests.post(POST_URL, headers=headers, data=ques_dict,
				timeout=(3,TIMEOUT)).content
		table_headers = ['Rank', 'Answer', 'Doc', 'Answer Score', 'Doc Score']
	#	return render_template()
		#print(predictions)
		predictions = json.loads(predictions)
		
		predictions = predictions['results']

		for i, p in enumerate(predictions, 1):
			answers.append([i, p['span'], p['doc_id'],
							'%.5g' % p['span_score'],
							'%.5g' % p['doc_score']])
		#print(answers)
		dict = {'scores':answers, 'ques':ques, 'headers': table_headers }

	return render_template('results.html',result = dict)

@app.route(BASE + '/results_new/<topn>/<retrieval_model>/<model>/<show_context>/')
@app.route(BASE + '/results_new/<topn>/<retrieval_model>/<model>/<show_context>/<ques>')
def results_new(topn, model, show_context, retrieval_model, ques=None):
	# model = model version id
	models_new = get_models_list()
	retrieval_models_new = get_retrieval_models_list()
	headers = {'Content-Type': 'application/json'}
	
	answers = []
	text = []
	output_text = []
	
	#print("abc")
	print("ip1:",request.remote_addr)
	print("url:",request.path)
	if ques is None:
		print("abcd")
		return redirect(url_for('index'))
	if model == "drqa" or model=="1":
		ques_dict = jsonify(question = ques)
		ques_dict = json.dumps({'question':ques, 'topn': topn, 'ip':request.remote_addr, 
			'url':request.path, 'model_version_id': model, 'retrieval_model_id': retrieval_model})

		results = requests.post(POST_URL, headers=headers, data=ques_dict,
				timeout=(3,TIMEOUT)).json()['results']
		#print(results)
		models = requests.get(MODEL_URL, timeout=(3,TIMEOUT)).json()
		#table_headers = ['Rank', 'Answer', 'Doc', 'Answer Score', 'Doc Score', 'Most Correct Answer']
		#return render_template()
		table_headers = []
		table_headers1 = requests.get(TABLE_HEADERS_URL + model, timeout=(3,TIMEOUT)).json()
		table_headers1 = sorted(table_headers1['table_headers'],key=lambda x: x[0])
		for i in table_headers1:
			table_headers.append(i[1])

		print(table_headers1)
		print(table_headers)
		print(models)
		#models = json.loads(models)
		
		#predictions = results['predictions']
		answers = results['answers']
		output_text = results['output_text']
		query_id = results['query_id']
		
		for j,output in enumerate(output_text):
			output_text[j] = Markup(output)

		dict = {'scores':answers, 'ques':ques, 'headers': table_headers,
				'show_context':show_context, 'output_text':output_text, 'query_id': query_id,
				'models': models_new, 'retrieval_models':retrieval_models_new}
		return render_template('results_new.html',result = dict)

	elif model=="2":
		RNET_URL = 'http://10.129.2.77:' + '9000' + '/api'
		ques_dict = json.dumps({'question': ques, 'topn': topn,
			'ip':request.remote_addr, 'url':request.path, 'model_version_id': model,
			'retrieval_model_id': retrieval_model})
		results = requests.post(POST_URL, headers=headers, data=ques_dict,
				timeout=(3,TIMEOUT)).json()['results']
		table_headers = []
		table_headers1 = requests.get(TABLE_HEADERS_URL + model, timeout=(3,TIMEOUT)).json()
		table_headers1 = sorted(table_headers1['table_headers'],key=lambda x: x[0])
		#print(table_headers)
		for i in table_headers1:
			#print(i,"\n\n\n")
			table_headers.append(i[1])
		answers = results['answers']
		output_text = results['output_text']
		query_id = results['query_id']

		for j,output in enumerate(output_text):
			output_text[j] = Markup(output)
		dict = {'scores':answers,'ques':ques, 'headers': table_headers, 'show_context':show_context,
				'output_text':output_text, 'query_id': query_id,
				'models': models_new, 'retrieval_models':retrieval_models_new}
		return render_template('results_new.html',result = dict)
	else:
		flash("No models found.")
		return redirect(url_for('index'))
def get_db():
	db = getattr(g, '_database', None)
	if db is None:
		db = g._database = sqlite3.connect(DATABASE)
	return db

@app.teardown_appcontext
def close_connection(exception):
	db = getattr(g, '_database', None)
	if db is not None:
		db.close()

# gets only models list which are enabled 
def get_models_list():
	models = requests.get(MODEL_URL, timeout=(3,TIMEOUT)).content
	models = json.loads(models)
	#print(models)
	models_new = []
	# i = [version_id, name, version_name]
	for i in models['models']:
		print(i)
		j = [i[0], i[1].strip() + " " + i[2].strip()]
		models_new.append(j)
	return models_new

# gets all models list which are enabled + disabled
def get_all_models_list():
	models = requests.get(ALL_MODEL_URL, timeout=(3,TIMEOUT)).content
	models = json.loads(models)
	#print(models)
	models_new = []
	# i = [version_id, name, version_name, flag]
	for i in models['models']:
		print(i)
		j = [i[0], i[1].strip() + " " + i[2].strip()]
		models_new.append(j)
	return models

# gets retrieval models list  
def get_retrieval_models_list():
	models = requests.get(RETRIEVAL_MODEL_URL, timeout=(3,TIMEOUT)).content
	models = json.loads(models)
	#print(models)
	models_new = []
	# i = [version_id, name, version_name]
	for i in models['models']:
		print(i)
		j = [i[0], i[1].strip()]
		models_new.append(j)
	return models_new

"""
@app.route(BASE + '/list_all_models')
def list_all_models():
	models = get_all_models_list()
	models_new = []
	for i in models['models']:
		j = {'mv_id':i[0], 'name':i[1], 'version':i[2], 'flag':i[3]}
		models_new.append(j)
	print(models_new)
	return render_template('list_models.html',result={'models':models_new})
"""

@app.route(BASE + '/toggle_model',methods=['POST'])
def toggle_model():
	headers = {'Content-Type': 'application/json'}

	mv_id = None
	mv_id_dict = {}
	URL = 'http://10.129.2.77:' + PORT
	for key in request.form.keys():
		flag = request.form[key]
		if flag == "Enable":
			URL += "/enable_model"
		elif flag == "Disable":
			URL += "/disable_model"
		mv_id_dict = {'model_version_id':int(key)}
		mv_id_dict = json.dumps(mv_id_dict)
	print(URL)
	results = requests.post(URL, headers=headers, data=mv_id_dict,
				timeout=(3,TIMEOUT))
	print(results)

	models = get_all_models_list()
	models_new = []
	for i in models['models']:
		j = {'mv_id':i[0], 'name':i[1], 'version':i[2], 'flag':i[3]}
		models_new.append(j)
	print(models_new)
	return render_template('list_models.html',result={'models':models_new})

@app.route(BASE)
def index():
	models_new = get_models_list()
	retrieval_models_new = get_retrieval_models_list()
	print('models_new:',models_new)
	return render_template('main_new.html',result={'models':models_new,
		'retrieval_models':retrieval_models_new})

@app.route(BASE + '/new')
def new_index():
	return render_template('main_new.html')

@app.route(BASE + '/more')
def index_more():
	return render_template('main_elaborate.html')

@app.route(BASE + '/query_log',methods = ['POST', 'GET'])
def query_log():
	models_new = get_models_list()
	retrieval_models_new = get_retrieval_models_list()
	return render_template('query_log.html',result={'models':models_new,
		'retrieval_models':retrieval_models_new})

@app.route(BASE + '/submit_feedback',methods=['POST'])
def submit_feedback():
	headers = {'Content-Type': 'application/json'}
	query_id = request.form['query_id']
	correct_ans_rank = request.form.get('optradio')
	print(query_id,correct_ans_rank)
	#conn = sqlite3.connect(DATABASE)
	dict = {'query_id': query_id, 'correct_ans_rank': correct_ans_rank}
	dict = json.dumps(dict)
	results = requests.post(FEEDBACK_URL, headers=headers, data=dict,
				timeout=(3,TIMEOUT))
	if correct_ans_rank is not None:
		flash("Thank you for your feedback.")
		return redirect(url_for('index'))
	else:
		flash("No answer selected. Please select an answer before submitting.")
		return redirect(url_for('index'))

@app.route('/extract',methods = ['POST', 'GET'])
def extract():
	if request.method == 'POST':
		question = request.form['ques']
		topn = request.form['topn']
		model = request.form['model']
		#print(question)
		return redirect(url_for('results', ques=question, topn=topn, model=model))
	else:
		question = request.args.get('ques')
		return redirect(url_for('results', ques=question, topn=topn, model=model))

@app.route('/extract_more',methods = ['POST', 'GET'])
def extract_more():
	if request.method == 'POST':
		question = request.form['ques']
		topn = request.form['topn']
		model = request.form['model']
		
		#print(question)
		return redirect(url_for('results_more', ques=question, topn=topn, model=model))
	else:
		question = request.args.get('ques')
		return redirect(url_for('results_more', ques=question, topn=topn, model=model))

@app.route(BASE + '/extract_new',methods = ['POST', 'GET'])
def extract_new():
	if request.method == 'POST':
		question = request.form['ques']
		topn = request.form['topn']
		model = request.form['model']
		#show_context  =request.form.getlist('context_check') # also correct way
		show_context  = request.form.get('context_check') != None
		retrieval_model = request.form['retrieval_model']
		print("question:",question)
		#print("show_context:",show_context)
		#print("show_context_type:",type(show_context))
		#show_context = True
		return redirect(url_for('results_new', topn=topn, model=model, show_context=show_context,
			retrieval_model=retrieval_model,ques=question))
	else:
		question = request.args.get('ques')
		return redirect(url_for('results_new', topn=topn, model=model, show_context=show_context,
			retrieval_model=retrieval_model,ques=question))

@app.route('/results_more/<ques>/<topn>')
def results_more(ques, topn):
	headers = {'Content-Type': 'application/json'}
	ques_dict = jsonify(question = ques)
	ques_dict = json.dumps({'question':ques, 'topn': topn})
	answers = []
	text = []
	predictions = requests.post(POST_URL, headers=headers, data=ques_dict,
			timeout=(3,TIMEOUT)).content
	predictions = json.loads(predictions)
	#print(predictions)
	predictions = predictions['results']

	for i, p in enumerate(predictions, 1):
		answers.append([i, p['span'], p['doc_id'],
						'%.5g' % p['span_score'],
						'%.5g' % p['doc_score']])
		text.append(p['context']['text'])
	#print(answers)
	#print(text)
	dict = {'scores':answers, 'ques':ques, 'text':text}

	return render_template('results_elaborate.html',result = dict)



@app.context_processor
def inject_enumerate():
	return dict(enumerate=enumerate)

if __name__ == '__main__':
	app.secret_key = 'super secret key'
	app.config['SESSION_TYPE'] = 'filesystem'
	app.run(host="0.0.0.0", port=9002, debug = True)