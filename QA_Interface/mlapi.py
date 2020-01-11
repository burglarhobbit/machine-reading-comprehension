#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Interactive interface to full DrQA pipeline."""

import torch
import argparse
import code
import prettytable
import logging

from termcolor import colored
from drqa import pipeline
from drqa.retriever import utils

import requests
from flask import Flask, abort, request, jsonify, g
import sqlite3
import json

import atexit

### imports for admin
import os
from flask import Flask, url_for, redirect, render_template, request
from flask_sqlalchemy import SQLAlchemy
from wtforms import form, fields, validators
import flask_login as login
import flask_admin as admin
from flask_admin.contrib import sqla
from flask_admin import helpers, expose
from werkzeug.security import generate_password_hash, check_password_hash

def exit_handler():
	global process_dict,retrieval_model_process_dict
	for p in process_dict.keys():
		
		process = process_dict[p]
		process.terminate()
		code = process.wait()
	for p in retrieval_model_process_dict.keys():
		process = retrieval_model_process_dict[p]
		process.terminate()
		code = process.wait()

app = Flask(__name__)

## following app configs are for admin
# Create dummy secrey key so we can use sessions
app.config['SECRET_KEY'] = '123456790'

# Create in-memory database
app.config['DATABASE_FILE'] = 'sample_db.sqlite'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + app.config['DATABASE_FILE']
app.config['SQLALCHEMY_ECHO'] = True
db = SQLAlchemy(app)

TIMEOUT = 60
# ------------------------------------------------------------------------------
# Drop in to interactive mode
# ------------------------------------------------------------------------------


def process(question, candidates=None, top_n=1, n_docs=5):
	global process_dict
	DrQA = process_dict.get("1",None)
	predictions = DrQA.process(
		question, candidates, top_n, n_docs, return_context=True
	)
	table = prettytable.PrettyTable(
		['Rank', 'Answer', 'Doc', 'Answer Score', 'Doc Score']
	)
	for i, p in enumerate(predictions, 1):
		table.add_row([i, p['span'], p['doc_id'],
					   '%.5g' % p['span_score'],
					   '%.5g' % p['doc_score']])
	print('Top Predictions:')
	print(table)
	print('\nContexts:')
	for p in predictions:
		text = p['context']['text']
		start = p['context']['start']
		end = p['context']['end']
		output = (text[:start] +
				  colored(text[start: end], 'green', attrs=['bold']) +
				  text[end:])
		print('[ Doc = %s ]' % p['doc_id'])
		print(output + '\n')
	return predictions

@app.route('/api',methods=['POST'])
def get_answer():
	from copy import deepcopy
	TIMEOUT = 45
	data = request.get_json(force=True)
	question = data['question']
	topn = int(data['topn'])
	ip = data['ip']
	url = data['url']
	model = data['model_version_id']
	retrieval_model = data['retrieval_model_id']
	answers = []
	output_text = []
	text = []
	font_color_start = '<font color="yellow">'
	font_color_end = '</font>'
	dict = {}
	if model == "1":
		headers = {'Content-Type': 'application/json'}
		URL = "http://10.129.2.77:9003/api"
		ques_dict = {'question':question,'topn':topn}
		ques_dict = json.dumps(ques_dict)
		#predictions = process(question, candidates=None, top_n=topn)
		predictions = requests.post(URL, headers=headers, data=ques_dict,
				timeout=(3,TIMEOUT)).json()['results']

		for i, p in enumerate(predictions, 1):
			answers.append([i, p['span'], p['doc_id'],
							'%.5g' % p['span_score'],
							'%.5g' % p['doc_score']])
			start = p['context']['start']
			end = p['context']['end']
			context = p['context']['text']
			output = context[:start] + font_color_start + context[start:end] + \
									font_color_end + context[end:]
			text.append(context)
			#output = Markup(output)
			#flash(output)
			output_text.append(output)
	
		predictions_copy = deepcopy(predictions)

		for i in range(int(topn)):
			start,end = predictions[i]['context']['start'],predictions[i]['context']['end']
			predictions_copy[i].pop('context',None)
			predictions_copy[i].pop('span',None)
			predictions_copy[i].pop('span_score',None)
			predictions_copy[i].pop('doc_score',None)
			predictions_copy[i]['answer'] = predictions[i]['context']['text'][start:end]
			predictions_copy[i]['rank'] = i+1 # rank starts with 1

		predictions_copy = {'results':predictions_copy}
		predictions_copy = json.dumps(predictions_copy)
		
		query_id = execute_query_log(ip, url, predictions_copy)
		dict = {'predictions':predictions, 'answers':answers, 'output_text': output_text, 
			'query_id': query_id}
	elif model == "2":
		headers = {'Content-Type': 'application/json'}
		RNET_URL = 'http://10.129.2.77:' + '9000' + '/api'
		RETRIEVAL_URL = 'http://10.129.2.77:' + '9004' + '/api'
		answers = []
		output_text = []
		predictions_rnet = [{} for i in range(topn)]
		ques_dict = {'question':question,'topn':topn}
		ques_dict = json.dumps(ques_dict)
		#predictions = process(question, candidates=None, top_n=topn)
		predictions = requests.post(RETRIEVAL_URL, headers=headers, data=ques_dict,
				timeout=(3,TIMEOUT)).json()['results']
		for i,p in enumerate(predictions):
			context = p['context']['text']
			data_dict = json.dumps({'context':context, 'question': question})
			results = requests.post(RNET_URL, headers=headers, data=data_dict,
				timeout=(3,TIMEOUT)).json()['results']
			start = results['start']
			end = results['end']

			output = context[:start] + font_color_start + context[start:end] + \
					 font_color_end + context[end:]

			answers.append([i,context[start:end], p['doc_id'],'%.5g' % p['doc_score']])
			output_text.append(output)

			predictions_rnet[i]['doc_id'] = p['doc_id']
			predictions_rnet[i]['answer'] = context[start:end]
			predictions_rnet[i]['rank'] = i+1
		"""
		context = "In meteorology, precipitation is any product of the condensation " \
			  "of atmospheric water vapor that falls under gravity. The main forms " \
			  "of precipitation include drizzle, rain, sleet, snow, graupel and hail." \
			  "Precipitation forms as smaller droplets coalesce via collision with other " \
			  "rain drops or ice crystals within a cloud. Short, intense periods of rain " \
			  "in scattered locations are called “showers”."
		data_dict = json.dumps({'context':context, 'question': question})
		results = requests.post(RNET_URL, headers=headers, data=data_dict,
				timeout=(3,TIMEOUT)).json()['results']
		print("\n\n\n",results,"\n\n\n")
		start = results['start']
		end = results['end']

		output = context[:start] + font_color_start + context[start:end] + font_color_end + \
									context[end:]

		answers.append([0,context[start:end]])
		output_text.append(output)
		# i = 0
		predictions[0]['answer'] = context[start:end]
		predictions[0]['rank'] = 1
		"""
		predictions = {'results':predictions_rnet}
		predictions = json.dumps(predictions)
		query_id = execute_query_log(ip, url, predictions)
		dict = {'answers':answers, 'output_text': output_text, 
			'query_id': query_id}
	return jsonify(results = dict)


DATABASE = 'flaskr/flaskr/nvli.db'

def execute_query_log(ip, url, predictions_copy):
	conn = get_db()
	cur = conn.cursor()

	cur.execute(
	'''
	INSERT INTO query_log (ip,query_url,response_obj) VALUES (?,?,?);
	''',(ip, url, predictions_copy))	
	conn.commit()
	#cur.execute('select * from query_log')
	#print(cur.fetchall())
	qid = cur.lastrowid
	return qid

def get_query_log():
	cur = get_db().cursor()
	cur.execute('''SELECT a.qid,a.ip,a.query_url,a.response_obj,b.correct_ans_rank,a.sqltime
					FROM query_log a
						LEFT JOIN feedback b
						ON a.qid = b.qid''')
	output = cur.fetchall()
	print(output)
	return output

def get_db():
	db = getattr(g, '_database', None)
	if db is None:
		db = g._database = sqlite3.connect(DATABASE)
	return db

@app.teardown_appcontext
def close_connection(exception):
	db = getattr(g, '_database', None)
	if db is not None:
		#db.close()
		pass

@app.route('/get_models',methods=['GET'])
def get_models():
	#conn = sqlite3.connect(DATABASE)
	cur = get_db().cursor()
	cur.execute('''SELECT model_version.mv_id,model.model_name,model_version.version
					FROM model_version
					INNER JOIN model ON model_version.model_id = model.model_id
					INNER JOIN activated_models ON activated_models.mv_id = model_version.mv_id
					WHERE activated_models.flag = 1''')
	output = cur.fetchall()
	print(output)
	return jsonify(models = output)


def get_all_models_list():
	cur = get_db().cursor()
	cur.execute('''SELECT model_version.mv_id,model.model_name,model_version.version,
					activated_models.flag		
					FROM model_version
					INNER JOIN model ON model_version.model_id = model.model_id
					INNER JOIN activated_models ON activated_models.mv_id = model_version.mv_id
				''')
	output = cur.fetchall()
	print(output)
	return output

def get_all_retrieval_models_list():
	cur = get_db().cursor()
	cur.execute('''SELECT retrieval_models.model_id,retrieval_models.model_name,
					activated_retrieval_models.flag
					FROM retrieval_models
					INNER JOIN activated_retrieval_models ON activated_retrieval_models.model_id = retrieval_models.model_id
				''')
	output = cur.fetchall()
	print(output)
	return output

@app.route('/get_all_models',methods=['GET'])
def get_all_models():
	#conn = sqlite3.connect(DATABASE)
	cur = get_db().cursor()
	cur.execute('''SELECT model_version.mv_id,model.model_name,model_version.version,
					activated_models.flag		
					FROM model_version
					INNER JOIN model ON model_version.model_id = model.model_id
					INNER JOIN activated_models ON activated_models.mv_id = model_version.mv_id
				''')
	output = cur.fetchall()
	print(output)
	return jsonify(models = output)

@app.route('/get_retrieval_models',methods=['GET'])
def get_retrieval_models():
	#conn = sqlite3.connect(DATABASE)
	cur = get_db().cursor()
	cur.execute('''SELECT retrieval_models.model_id,retrieval_models.model_name
					FROM retrieval_models
					INNER JOIN activated_retrieval_models ON activated_retrieval_models.model_id = retrieval_models.model_id
					WHERE activated_retrieval_models.flag = 1''')
	output = cur.fetchall()
	print(output)
	return jsonify(models = output)

@app.route('/get_headers/<modelversion_id>',methods=['GET'])
def get_headers(modelversion_id):
	#conn = sqlite3.connect(DATABASE)
	cur = get_db().cursor()
	cur.execute('''SELECT rank, param
					FROM parameters
					WHERE parameters.mv_id = ? 
					ORDER BY parameters.mv_id ASC''',modelversion_id)
	output = cur.fetchall()
	print(output)
	return jsonify(table_headers = output)

@app.route('/submit_feedback',methods=['POST'])
def submit_feedback():
	data = request.get_json(force=True)
	#conn = sqlite3.connect(DATABASE)
	qid = data['query_id']
	correct_ans_rank = data['correct_ans_rank']
	conn = get_db()
	cur = conn.cursor()
	cur.execute('INSERT INTO feedback (qid,correct_ans_rank) VALUES (?,?)',(qid,correct_ans_rank))
	conn.commit()
	cur.execute('select * from feedback')
	print(cur.fetchall())
	return "ok",200

@app.route('/query_log',methods=['POST'])
def query_log():
	return 200
"""
@app.route('/enable_model',methods=['POST'])
def enable_model():
	global process_dict
	from subprocess import Popen

	data = request.get_json(force=True)
	model_version_id = data['model_version_id']
	
	conn = get_db()
	cur = conn.cursor()
	cur.execute(
		'''
		UPDATE activated_models SET flag=1 WHERE mv_id=?;
		''',(model_version_id,))
	conn.commit()
	#cur.execute('select * from query_log')
	#print(cur.fetchall())
	qid = cur.lastrowid

	if model_version_id==1:
		p = Popen(["python", "/home/search/DrQA/scripts/pipeline/mlapi_bhavya.py","--port","9003"])
		process_dict["1"] = p
	elif model_version_id==2:
		p = Popen(["python", "/home/search/DrQA/scripts/pipeline/rnet/R-Net_new/inference.py"])
		process_dict["2"] = p
	return "ok",200

@app.route('/disable_model',methods=['POST'])
def disable_model():
	global process_dict
	from subprocess import Popen
	
	data = request.get_json(force=True)
	model_version_id = data['model_version_id']

	conn = get_db()
	cur = conn.cursor()
	cur.execute(
		'''
		UPDATE activated_models SET flag=0 WHERE mv_id=?;
		''',(model_version_id,))
	conn.commit()
	#cur.execute('select * from query_log')
	#print(cur.fetchall())
	qid = cur.lastrowid

	if model_version_id==1:
		p = process_dict.pop("1", None)
		if p is not None:
			p.terminate()
			returncode = p.wait()
			print("Returncode of subprocess: %s" % returncode)
	elif model_version_id==2:
		p = process_dict.pop("2", None)
		if p is not None:
			p.terminate()
			returncode = p.wait()
			print("Returncode of subprocess: %s" % returncode)
	return "ok",200
"""
###############################
# for admin:
###############################

# Create user model.
class User(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	first_name = db.Column(db.String(100))
	last_name = db.Column(db.String(100))
	login = db.Column(db.String(80), unique=True)
	email = db.Column(db.String(120))
	password = db.Column(db.String(64))

	# Flask-Login integration
	def is_authenticated(self):
		return True

	def is_active(self):
		return True

	def is_anonymous(self):
		return False

	def get_id(self):
		return self.id

	# Required for administrative interface
	def __unicode__(self):
		return self.username


# Define login and registration forms (for flask-login)
class LoginForm(form.Form):
	login = fields.TextField(validators=[validators.required()])
	password = fields.PasswordField(validators=[validators.required()])

	def validate_login(self, field):
		user = self.get_user()

		if user is None:
			raise validators.ValidationError('Invalid user')

		# we're comparing the plaintext pw with the the hash from the db
		if not check_password_hash(user.password, self.password.data):
		# to compare plain text passwords use
		# if user.password != self.password.data:
			raise validators.ValidationError('Invalid password')

	def get_user(self):
		return db.session.query(User).filter_by(login=self.login.data).first()


class RegistrationForm(form.Form):
	login = fields.TextField(validators=[validators.required()])
	email = fields.TextField()
	password = fields.PasswordField(validators=[validators.required()])

	def validate_login(self, field):
		if db.session.query(User).filter_by(login=self.login.data).count() > 0:
			raise validators.ValidationError('Duplicate username')


# Initialize flask-login
def init_login():
	login_manager = login.LoginManager()
	login_manager.init_app(app)

	# Create user loader function
	@login_manager.user_loader
	def load_user(user_id):
		return db.session.query(User).get(user_id)


# Create customized model view class
class MyModelView(sqla.ModelView):

	def is_accessible(self):
		return login.current_user.is_authenticated


# Create customized index view class that handles login & registration
class MyAdminIndexView(admin.AdminIndexView):

	@expose('/')
	def index(self):
		if not login.current_user.is_authenticated:
			return redirect(url_for('.login_view'))

		return self.render('admin/index.html')
		#return super(MyAdminIndexView, self).index()

	@expose('/login/', methods=('GET', 'POST'))
	def login_view(self):
		# handle user login
		form = LoginForm(request.form)
		if helpers.validate_form_on_submit(form):
			user = form.get_user()
			login.login_user(user)

		if login.current_user.is_authenticated:
			return redirect(url_for('.index'))
		link = '<p>Don\'t have an account? <a href="' + url_for('.register_view') + '">Click here to register.</a></p>'
		self._template_args['form'] = form
		self._template_args['link'] = link
		return self.render('admin/index.html')
		#return super(MyAdminIndexView, self).index()

	@expose('/register/', methods=('GET', 'POST'))
	def register_view(self):
		form = RegistrationForm(request.form)
		if helpers.validate_form_on_submit(form):
			user = User()

			form.populate_obj(user)
			# we hash the users password to avoid saving it as plaintext in the db,
			# remove to use plain text:
			user.password = generate_password_hash(form.password.data)

			db.session.add(user)
			db.session.commit()

			login.login_user(user)
			return redirect(url_for('.index'))
		link = '<p>Already have an account? <a href="' + url_for('.login_view') + '">Click here to log in.</a></p>'
		self._template_args['form'] = form
		self._template_args['link'] = link
		return self.render('admin/index.html')
		#return super(MyAdminIndexView, self).index()

	@expose('/list_all_models', methods=('GET','POST'))
	def list_all_models_view(self):
		if login.current_user.is_authenticated and login.current_user.login == "admin":
			models = get_all_models_list()
			retrieval_models = get_all_retrieval_models_list()
			models_new = []
			retrieval_models_new = []
			for i in models:
				j = {'mv_id':i[0], 'name':i[1], 'version':i[2], 'flag':i[3]}
				models_new.append(j)
			for i in retrieval_models:
				j = {'model_id':i[0], 'name':i[1], 'flag':i[2]}
				retrieval_models_new.append(j)
			print(models_new)
			#return self.render('admin/index.html',result={'models':models_new})
			return self.render('admin/index_old.html',result={'models':models_new,'retrieval_models':
				retrieval_models_new})
		return redirect(url_for('.login_view'))

	@expose('/toggle_model', methods=['GET','POST'])
	def toggle_model(self):
		headers = {'Content-Type': 'application/json'}
		print(login.current_user.is_authenticated,login.current_user.login)
		if login.current_user.is_authenticated and login.current_user.login == "admin":
			data_dict = {}
			URL = 'http://10.129.2.77:9001/admin'
			results = None
			flag = None
			# key is model_version_id for models or model_id for retrieval_models
			for key in request.form.keys():
				if "model_type" in key:
					continue
				flag = request.form[key]
				if flag == "Enable":
					URL += "/enable_model"
				elif flag == "Disable":
					URL += "/disable_model"
				# search model vs retrieval model
				if request.form["model_type"] == "model":
					mv_id_dict = {'model_version_id':int(key),'model_type':'model'}
					data_dict = json.dumps(mv_id_dict)
					data = mv_id_dict
				elif request.form["model_type"] == "retrieval_model":
					model_id_dict = {'model_id':int(key),'model_type':'retrieval_model'}
					data_dict = json.dumps(model_id_dict)
					data = model_id_dict
			print(URL)
			#results = requests.post(URL, headers=headers, data=data_dict,
			#			timeout=(3,TIMEOUT))
			if flag == "Enable":
				results = self.enable_model(data)
			elif flag == "Disable":
				results = self.disable_model(data)
			print(results)

			models = get_all_models_list()
			retrieval_models = get_all_retrieval_models_list()
			models_new = []
			retrieval_models_new = []
			for i in models:
				j = {'mv_id':i[0], 'name':i[1], 'version':i[2], 'flag':i[3]}
				models_new.append(j)
			for i in retrieval_models:
				j = {'model_id':i[0], 'name':i[1], 'flag':i[2]}
				retrieval_models_new.append(j)
			print(models_new)
			return redirect(url_for('.list_all_models_view'))
			return self.render('admin/index_old.html',result={'models':models_new,'retrieval_models':
				retrieval_models_new})
		
	#@expose('/enable_model', methods=['GET','POST'])
	def enable_model(self,data):
		global process_dict,retrieval_model_process_dict
		from subprocess import Popen
		if login.current_user.is_authenticated and login.current_user.login == "admin":
			#data = request.get_json(force=True)
			
			if data['model_type'] == 'model':
				model_version_id = data['model_version_id']
			
				conn = get_db()
				cur = conn.cursor()
				cur.execute(
					'''
					UPDATE activated_models SET flag=1 WHERE mv_id=?;
					''',(model_version_id,))
				conn.commit()
				#cur.execute('select * from query_log')
				#print(cur.fetchall())
				qid = cur.lastrowid
				if model_version_id==1:
					p = Popen(["python", "/home/search/DrQA/scripts/pipeline/mlapi_bhavya.py","--port","9003"])
					process_dict["1"] = p
				elif model_version_id==2:
					p = Popen(["python", "/home/search/DrQA/scripts/pipeline/rnet/R-Net_new/inference.py"])
					process_dict["2"] = p
			elif data['model_type'] == 'retrieval_model':
				model_id = data['model_id']
			
				conn = get_db()
				cur = conn.cursor()
				cur.execute(
					'''
					UPDATE activated_retrieval_models SET flag=1 WHERE model_id=?;
					''',(model_id,))
				conn.commit()
				#cur.execute('select * from query_log')
				#print(cur.fetchall())
				qid = cur.lastrowid
				if model_id==1:
					p = Popen(["python", "/home/search/DrQA/scripts/pipeline/mlapi_bhavya.py","--port","9004"])
					retrieval_model_process_dict["1"] = p
			return "ok",200

	#@expose('/disable_model', methods=['GET','POST'])
	def disable_model(self,data):
		global process_dict,retrieval_model_process_dict
		from subprocess import Popen
		#print("\n\n::::enter_function::::")
		#print(login.current_user.is_authenticated,login.current_user.login)
		if login.current_user.is_authenticated and login.current_user.login == "admin":
			#print("\n\n::::is_authenticated::::\n\n")
			#data = request.get_json(force=True)
			if data["model_type"] == "model":
				#print("\n\n::::model::::\n\n")
				model_version_id = data['model_version_id']

				conn = get_db()
				cur = conn.cursor()
				cur.execute(
					'''
					UPDATE activated_models SET flag=0 WHERE mv_id=?;
					''',(model_version_id,))
				conn.commit()
				#cur.execute('select * from query_log')
				#print(cur.fetchall())
				qid = cur.lastrowid

				if model_version_id==1:
					p = process_dict.pop("1", None)
					if p is not None:
						p.terminate()
						returncode = p.wait()
						print("Returncode of subprocess: %s" % returncode)
				elif model_version_id==2:
					p = process_dict.pop("2", None)
					if p is not None:
						p.terminate()
						returncode = p.wait()
						print("Returncode of subprocess: %s" % returncode)
			elif data["model_type"] == "retrieval_model":
				#print("\n\n::::retrieval_model::::\n\n")
				model_id = data['model_id']

				conn = get_db()
				cur = conn.cursor()
				cur.execute(
					'''
					UPDATE activated_retrieval_models SET flag=0 WHERE model_id=?;
					''',(model_id,))
				conn.commit()
				#cur.execute('select * from query_log')
				#print(cur.fetchall())
				qid = cur.lastrowid

				if model_id==1:
					p = retrieval_model_process_dict.pop("1", None)
					if p is not None:
						p.terminate()
						returncode = p.wait()
						print("Returncode of subprocess: %s" % returncode)
			else:
				print("\n\n::::NONE::::\n\n")
			return "ok",200
		else:
			print("\n\n::::NONE::::\n\n")

	@expose('/query_log', methods=['GET','POST'])
	def query_log(self):
		if login.current_user.is_authenticated and login.current_user.login == "admin":
			output = get_query_log()
			return self.render('admin/query_log.html',result={'output':output})
		return redirect(url_for('.login_view'))
	@expose('/logout/')
	def logout_view(self):
		login.logout_user()
		return redirect(url_for('.index'))


# Flask views
@app.route('/')
def index():
	return render_template('index.html')


# Initialize flask-login
init_login()

# Create admin
admin = admin.Admin(app, 'Example: Auth', index_view=MyAdminIndexView(), base_template='my_master.html')

# Add view
admin.add_view(MyModelView(User, db.session))


def build_sample_db():
	"""
	Populate a small db with some example entries.
	"""

	import string
	import random

	db.drop_all()
	db.create_all()
	# passwords are hashed, to use plaintext passwords instead:
	# test_user = User(login="test", password="test")
	test_user = User(login="admin", password=generate_password_hash("admin"))
	test_user.first_name = "Administrator"
	test_user.last_name = "Administrator"
	db.session.add(test_user)

	first_names = [
		'Harry', 'Amelia', 'Oliver', 'Jack', 'Isabella', 'Charlie','Sophie', 'Mia',
		'Jacob', 'Thomas', 'Emily', 'Lily', 'Ava', 'Isla', 'Alfie', 'Olivia', 'Jessica',
		'Riley', 'William', 'James', 'Geoffrey', 'Lisa', 'Benjamin', 'Stacey', 'Lucy'
	]
	last_names = [
		'Brown', 'Smith', 'Patel', 'Jones', 'Williams', 'Johnson', 'Taylor', 'Thomas',
		'Roberts', 'Khan', 'Lewis', 'Jackson', 'Clarke', 'James', 'Phillips', 'Wilson',
		'Ali', 'Mason', 'Mitchell', 'Rose', 'Davis', 'Davies', 'Rodriguez', 'Cox', 'Alexander'
	]

	for i in range(len(first_names)):
		user = User()
		user.first_name = first_names[i]
		user.last_name = last_names[i]
		user.login = user.first_name.lower()
		user.email = user.login + "@example.com"
		user.password = generate_password_hash(''.join(random.choice(string.ascii_lowercase + string.digits) for i in range(10)))
		#db.session.add(user)

	db.session.commit()
	return

@app.context_processor
def inject_enumerate():
	return dict(enumerate=enumerate)

if __name__ == '__main__':
	from subprocess import Popen
	global process_dict, retrieval_model_process_dict

	process_dict = {}
	retrieval_model_process_dict = {}

	with app.app_context():	
		models = get_all_models_list()
		retrieval_models = get_all_retrieval_models_list()
	for i in models:
		j = {'mv_id':i[0], 'name':i[1], 'version':i[2], 'flag':i[3]}
		if j['flag'] == 1 and i[0]==1:
			p = Popen(["python", "/home/search/DrQA/scripts/pipeline/mlapi_bhavya.py","--port","9003"])
			process_dict["1"] = p
		if j['flag'] == 1 and i[0]==2:
			p = Popen(["python", "/home/search/DrQA/scripts/pipeline/rnet/R-Net_new/inference.py"])
			process_dict["2"] = p
	for i in retrieval_models:
		j = {'model_id':i[0], 'name':i[1], 'flag':i[2]}
		if j['flag'] == 1 and i[0]==1:
			p = Popen(["python", "/home/search/DrQA/scripts/pipeline/mlapi_bhavya.py","--port","9004"])
			retrieval_model_process_dict["1"] = p

	# Build a sample db on the fly, if one does not exist yet.
	app_dir = os.path.realpath(os.path.dirname(__file__))
	database_path = os.path.join(app_dir, app.config['DATABASE_FILE'])
	if not os.path.exists(database_path):
		build_sample_db()

	app.run(host= "0.0.0.0",port=9001,debug=True)
#code.interact(banner=banner, local=locals())
