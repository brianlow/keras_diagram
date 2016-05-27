docker build -t keras_diagram . && \
docker run --rm -i -t -v `pwd`:/keras_diagram keras_diagram /bin/bash
