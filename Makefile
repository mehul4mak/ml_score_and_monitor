all: ingest train score deploy diagnose
	
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


