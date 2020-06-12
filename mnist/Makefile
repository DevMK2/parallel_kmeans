SHELL := /bin/bash

all: create_autoencoder_env autoencoder 

create_autoencoder_env:
	python3 -m venv keras-env &&\
		source keras-env/bin/activate &&\
		pip install --upgrade pip &&\
		pip install numpy &&\
		pip install scipy &&\
		pip install pandas &&\
		pip install seaborn &&\
		pip install matplotlib &&\
		pip install mkl &&\
		pip install sklearn &&\
		pip install --upgrade tensorflow &&\
		pip install --upgrade keras

autoencoder:
	mkdir mnist_encoded &&\
	source keras-env/bin/activate &&\
	python3 encoder/autoencoder.py

clean:
	rm -rf keras-env &&\
		rm -rf mnist_encoded
