Keras Diagram
=============

Print ASCII diagrams of your Keras models to visualize
the layers and their shapes.

          InputLayer (None, 50, 300)
             Reshape (None, 1, 50, 300)
       Convolution2D (None, 250, 48, 1)
                Relu (None, 250, 48, 1)
        MaxPooling2D (None, 250, 1, 1)
             Flatten (None, 250)
             Dropout (None, 250)
               Dense (None, 7)
             Softmax (None, 7)


To install
----------

    pip install keras_diagram


To use
------

	from keras_diagram import ascii

	model = Sequential()
	model.add(...)

	print ascii(model)

