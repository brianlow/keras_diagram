docker build -t keras_diagram . && \
docker run --rm --interactive -t keras_diagram /bin/bash -c "python setup.py sdist; python setup.py bdist_wheel; twine upload dist/*"

