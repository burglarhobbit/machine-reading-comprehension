import sqlite3
import datetime
from flask import g

models = [(1,'DrQA'),(2,'R-NET')] # (Model_id, name)
model_versions = [(1,1,'1.0'),(2,2,'1.0')] # (model_version_id, model_id, version)
parameters = [(1,'Rank',1),
			  (1,'Answer',2),
			  (1,'Doc', 3),
			  (1,'Answer Score',4),
			  (1,'Doc Score',5),
			  (1,'Select Most Correct Answer',6),
			  (2,'Rank',1),
			  (2,'Answer',2),
			  (2,'Doc', 3),
			  (2,'Doc Score',4),
			  (2,'Select Most Correct Answer',5),] # (model_version_id, parameter, parameter_rank) 

# if a model is enabled or not
enabled_disabed = [(1,1),
				   (2,1),
					] # (model_version_id, flag) 
					# 1 if flag = true,  enabled, 
					# 0 if flag = false, disabled

# if a model is enabled or not
enabled_disabed_retrieval = [(1,1),
					] # (model_version_id, flag) 
					# 1 if flag = true,  enabled, 
					# 0 if flag = false, disabled

retrieval_models = [(1,'DrQA')] # model_id, model_name
conn = sqlite3.connect('nvli.db')
c = conn.cursor()

c.execute('DROP TABLE query_log')
c.execute('DROP TABLE parameters')
c.execute('DROP TABLE model')
c.execute('DROP TABLE model_version')
c.execute('DROP TABLE feedback')
c.execute('DROP TABLE retrieval_models')
c.execute('DROP TABLE activated_models')
c.execute('DROP TABLE activated_retrieval_models')

c.execute('''CREATE TABLE query_log
			(qid INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
			 ip VARCHAR(15) NOT NULL,
			 query_url TEXT,
			 response_obj TEXT,
			 sqltime TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL);''')

c.execute('''CREATE TABLE parameters
			(mv_id INTEGER NOT NULL,
			 param VARCHAR(15) NOT NULL,
			 rank INTEGER NOT NULL,
			 PRIMARY KEY (mv_id, param));''')

c.execute('''CREATE TABLE model
			 (model_id INTEGER NOT NULL,
			 model_name VARCHAR(20) NOT NULL);''')

c.execute('''CREATE TABLE model_version
			(mv_id INTEGER PRIMARY KEY NOT NULL,
			 model_id INTEGER NOT NULL,
			 version VARCHAR(10) NOT NULL);''')

c.execute('''CREATE TABLE feedback
			(qid INTEGER PRIMARY KEY NOT NULL,
			 correct_ans_rank INTEGER NOT NULL);''')

c.execute('''CREATE TABLE retrieval_models
			(model_id INTEGER PRIMARY KEY NOT NULL,
			 model_name VARCHAR(20) NOT NULL);''')

c.execute('''CREATE TABLE activated_models
			(mv_id INTEGER PRIMARY KEY NOT NULL,
			 flag INTEGER NOT NULL);''')

c.execute('''CREATE TABLE activated_retrieval_models
			(model_id INTEGER PRIMARY KEY NOT NULL,
			 flag INTEGER NOT NULL);''')

c.execute('''
		INSERT INTO query_log (ip,query_url,response_obj) VALUES ('1.1.1.1','test/url','response_obj');
		''')

c.executemany('INSERT INTO model (model_id,model_name) VALUES (?,?)',models)
c.executemany('INSERT INTO model_version (mv_id,model_id,version) VALUES (?,?,?)',model_versions)
c.executemany('INSERT INTO parameters (mv_id,param,rank) VALUES (?,?,?)',parameters)
c.executemany('INSERT INTO feedback (qid,correct_ans_rank) VALUES (?,?)',[(3,5)])
c.executemany('INSERT INTO retrieval_models (model_id,model_name) VALUES (?,?)',retrieval_models)
c.executemany('INSERT INTO activated_models (mv_id,flag) VALUES (?,?)',enabled_disabed)
c.executemany('INSERT INTO activated_retrieval_models (model_id,flag) VALUES (?,?)',enabled_disabed_retrieval)

c.execute('select * from query_log')
a = c.fetchall()
c.execute('select param from parameters order by rank asc')
b = c.fetchall()
c.execute('select * from model_version')
d = c.fetchall()
conn.commit()
print(a,b,d)