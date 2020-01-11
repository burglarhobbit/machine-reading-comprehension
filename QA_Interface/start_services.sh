cd ~
source activate drqa

nc -z 10.129.2.77 8983 && echo "Core ML model is already online" || nohup /home/search/CDAC/solr-7.0.1/bin/solr start -p 8983 &

cd ~/DrQA/scripts/pipeline
nc -z 10.129.2.77 9001 && echo "ML backend db + API is already online" || nohup python3 mlapi.py

cd flaskr/flaskr
nc -z 10.129.2.77 9002 && echo "Frontend UI/UX services are already online" || nohup python3 flaskr.py
#echo `ls`