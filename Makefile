all: ingest train score deploy diagnose report 
	
ingest:
	python ingestion.py

train:
	python training.py

score:
	python scoring.py

deploy:
	python deployment.py

diagnose:
	python diagnostics.py

report:
	python reporting.py

fullprocess:
	python fullprocess.py

wsgi:
	python wsgi.py

apicall:
	python apicalls.py

app:
	python app.py


