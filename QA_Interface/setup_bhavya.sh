############################################################
# if solr is not running
nohup /home/search/CDAC/solr-7.0.1/bin/solr start -p 8983 &

############################################################
# run the following commands to start the pipeline

1. source activate drqa

2. cd DrQA

************************

3. export CLASSPATH=$CLASSPATH:/home/search/DrQA/stanford-corenlp-full-2017-06-09/*

************************
#4 is not needed
4. python3 scripts/pipeline/interactive.py  --no-cuda

#############################################################
Bhavya's edition:

# first time only:
cd scripts/pipeline/flaskr/flaskr
python3 database.py

# to run everytime 
source activate drqa
cd scripts/pipeline
python3 mlapi.py
cd scripts/pipeline/flaskr/flaskr
python3 flaskr.py

rnet would require tensorflow 1.3.0 or newer depending on the code version we're using

