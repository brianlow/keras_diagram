docker build -t keras_diagram . && \
docker run --rm -it keras_diagram python keras_diagram/diagram_test.py $1
