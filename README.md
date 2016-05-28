Keras Diagram
=============

Print ASCII diagrams of your [Keras](https://github.com/fchollet/keras) models to visualize the layers and their shapes.

A simple example:

          InputLayer (None, 50, 300)
             Reshape (None, 1, 50, 300)
       Convolution2D (None, 250, 48, 1)
                Relu (None, 250, 48, 1)
        MaxPooling2D (None, 250, 1, 1)
             Flatten (None, 250)
             Dropout (None, 250)
               Dense (None, 7)
             Softmax (None, 7)

A more complex model from [babi_memnn.py](https://github.com/fchollet/keras/blob/e2fb8b2786817b4014c077c13e99efb551fe35c1/examples/babi_memnn.py):

    InputLayer (None, 68)          InputLayer (None, 4)
     Embedding (None, 68, 64)       Embedding (None, 4, 64)
       Dropout (None, 68, 64)         Dropout (None, 4, 64)
               \____________________________/                     InputLayer (None, 68)
                             |                                     Embedding (None, 68, 4)
                        Merge (None, 68, 4)                          Dropout (None, 68, 4)
                              \____________________________________________/
                                             |                                                   InputLayer (None, 4)
                                        Merge (None, 68, 4)                                       Embedding (None, 4, 64)
                                      Permute (None, 4, 68)                                         Dropout (None, 4, 64)
                                              \___________________________________________________________/
                                                            |
                                                       Merge (None, 4, 132)
                                                        LSTM (None, 32)
                                                     Dropout (None, 32)
                                                       Dense (None, 22)
                                                     Softmax (None, 22)


To install
----------

    pip install keras_diagram


To use
------

    from keras_diagram import ascii

    model = Sequential()
    model.add(...)

    print(ascii(model))


Developing
------

     ./test.py       # run tests via docker
     ./publish.py    # build distributions and publish to pypi
     ./shell.py      # run bash above docker container with current folder mounted
