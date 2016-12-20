Keras Diagram
=============

Print ASCII diagrams of your [Keras](https://github.com/fchollet/keras) models to visualize the layers and their shapes.


          InputLayer (None, 50, 300)
             Reshape (None, 1, 50, 300)
       Convolution2D (None, 250, 48, 1)
                Relu (None, 250, 48, 1)
        MaxPooling2D (None, 250, 1, 1)
             Flatten (None, 250)
             Dropout (None, 250)
               Dense (None, 7)
             Softmax (None, 7)

A more complex model from [babi_rnn.py](https://github.com/fchollet/keras/blob/e2fb8b2786817b4014c077c13e99efb551fe35c1/examples/babi_rnn.py):

                                       InputLayer (None, 5)
                                        Embedding (None, 5, 50)
      InputLayer (None, 552)              Dropout (None, 5, 50)
       Embedding (None, 552, 50)             LSTM (None, 50)
         Dropout (None, 552, 50)     RepeatVector (None, 552, 50)
                 \______________________________/
                                |
                           Merge (None, 552, 50)
                            LSTM (None, 50)
                         Dropout (None, 50)
                           Dense (None, 36)


To install
----------

    pip install keras_diagram
    
    Pre-requisite in Conda Installation (Python 3.5): pip install pypandoc


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
