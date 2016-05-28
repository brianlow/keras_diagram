import unittest
from difflib import ndiff
from diagram import *
from keras.models import *
from keras.layers.core import *
from keras.layers.embeddings import *
from keras.layers.convolutional import *
from keras.layers import LSTM

class DiagramTest(unittest.TestCase):

	def test_simple(self):
		model = Sequential()
		model.add(Reshape((1, 50, 300), input_shape=(50, 300)))
		model.add(Convolution2D(250, 3, 300))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(50 - 3 + 1, 1)))
		model.add(Flatten())
		model.add(Dropout(0.5))
		model.add(Dense(7))
		expected = ("         InputLayer (None, 50, 300)    " + "\n"
					"            Reshape (None, 1, 50, 300) " + "\n"
					"      Convolution2D (None, 250, 48, 1) " + "\n"
					"               Relu (None, 250, 48, 1) " + "\n"
					"       MaxPooling2D (None, 250, 1, 1)  " + "\n"
					"            Flatten (None, 250)        " + "\n"
					"            Dropout (None, 250)        " + "\n"
					"              Dense (None, 7)          " + "\n"
				   )
		actual = ascii(model)
		self.assertStringsEqual(actual, expected)

	def test_simple_merge(self):
		left = Sequential()
		left.add(Reshape((1, 1, 1), input_shape=(1,1)))
		right = Sequential()
		right.add(Reshape((1, 1, 1), input_shape=(1,1)))
		model = Sequential()
		model.add(Merge([left, right], mode='concat', concat_axis=1))
		expected = ("      InputLayer (None, 1, 1)          InputLayer (None, 1, 1)    " + "\n"
					"         Reshape (None, 1, 1, 1)          Reshape (None, 1, 1, 1) " + "\n"
					"                 \______________________________/                 " + "\n"
					"                                |                                 " + "\n"
					"                           Merge (None, 2, 1, 1)                  " + "\n"
				   )
		actual = ascii(model)
		self.assertStringsEqual(actual, expected)

	def test_simple_fan_in_three(self):
		one = Sequential()
		one.add(Reshape((1, 1, 1), input_shape=(1,1)))
		two = Sequential()
		two.add(Reshape((1, 1, 1), input_shape=(1,1)))
		three = Sequential()
		three.add(Reshape((1, 1, 1), input_shape=(1,1)))
		model = Sequential()
		model.add(Merge([one, two, three], mode='concat', concat_axis=1))
		expected = ("      InputLayer (None, 1, 1)          InputLayer (None, 1, 1)          InputLayer (None, 1, 1)    " + "\n"
					"         Reshape (None, 1, 1, 1)          Reshape (None, 1, 1, 1)          Reshape (None, 1, 1, 1) " + "\n"
					"                 \_______________________________|_______________________________/                 " + "\n"
					"                                                 |                                                 " + "\n"
					"                                            Merge (None, 3, 1, 1)                                  " + "\n"
				   )
		actual = ascii(model)
		self.assertStringsEqual(actual, expected)

	def test_babi_memnn(self):
		input_encoder_m = Sequential()
		input_encoder_m.add(Embedding(input_dim=22, output_dim=64, input_length=68))
		input_encoder_m.add(Dropout(0.3))

		question_encoder = Sequential()
		question_encoder.add(Embedding(input_dim=22, output_dim=64, input_length=4))
		question_encoder.add(Dropout(0.3))

		match = Sequential()
		match.add(Merge([input_encoder_m, question_encoder], mode='dot', dot_axes=[2, 2]))

		input_encoder_c = Sequential()
		input_encoder_c.add(Embedding(input_dim=22, output_dim=4, input_length=68))
		input_encoder_c.add(Dropout(0.3))

		response = Sequential()
		response.add(Merge([match, input_encoder_c], mode='sum'))
		response.add(Permute((2, 1)))

		answer = Sequential()
		answer.add(Merge([response, question_encoder], mode='concat', concat_axis=-1))
		answer.add(LSTM(32))
		answer.add(Dropout(0.3))
		answer.add(Dense(22))
		answer.add(Activation('softmax'))

		expected = ("     InputLayer (None, 68)          InputLayer (None, 4)                                                                    " + "\n"
					"      Embedding (None, 68, 64)       Embedding (None, 4, 64)                                                                " + "\n"
					"        Dropout (None, 68, 64)         Dropout (None, 4, 64)                                                                " + "\n"
					"                \____________________________/                     InputLayer (None, 68)                                    " + "\n"
					"                              |                                     Embedding (None, 68, 4)                                 " + "\n"
					"                         Merge (None, 68, 4)                          Dropout (None, 68, 4)                                 " + "\n"
					"                               \____________________________________________/                                               " + "\n"
					"                                              |                                                   InputLayer (None, 4)      " + "\n"
					"                                         Merge (None, 68, 4)                                       Embedding (None, 4, 64)  " + "\n"
					"                                       Permute (None, 4, 68)                                         Dropout (None, 4, 64)  " + "\n"
					"                                               \___________________________________________________________/                " + "\n"
					"                                                             |                                                              " + "\n"
					"                                                        Merge (None, 4, 132)                                                " + "\n"
					"                                                         LSTM (None, 32)                                                    " + "\n"
					"                                                      Dropout (None, 32)                                                    " + "\n"
					"                                                        Dense (None, 22)                                                    " + "\n"
					"                                                      Softmax (None, 22)                                                    " + "\n"
				   )

		actual = ascii(answer)
		self.assertStringsEqual(actual, expected)


	def assertStringsEqual(self, s1, s2):
		if s1 != s2:
			diff = ndiff(s1.splitlines(1), s2.splitlines(1))
			print(s1)
			print
			print(''.join(diff))
			self.assertEqual(s1, s2, msg='Strings a different')

class ArrowsTest(unittest.TestCase):

    def test_draw_vertical(self):
    	a = Arrows(10)
    	a.draw(5, 5)
        self.assertEqual(a.line1.value, '     |    ')
        self.assertEqual(a.line2.value, '     |    ')

    def test_draw_left(self):
    	a = Arrows(10)
    	a.draw(3, 7)
        self.assertEqual(a.line1.value, r'    \___  ')
        self.assertEqual(a.line2.value, r'       |  ')

    def test_draw_right(self):
    	a = Arrows(10)
    	a.draw(7, 3)
        self.assertEqual(a.line1.value, r'   ___/   ')
        self.assertEqual(a.line2.value, r'   |      ')

    def test_draw_multiple_right(self):
    	a = Arrows(10)
    	a.draw(9, 0)
    	a.draw(6, 0)
        self.assertEqual(a.line1.value, r'_____/__/ ')
        self.assertEqual(a.line2.value, r'|         ')

    def test_draw_two(self):
    	a = Arrows(10)
    	a.draw(2, 5)
    	a.draw(8, 5)
        self.assertEqual(a.line1.value, r'   \___/  ')
        self.assertEqual(a.line2.value, r'     |    ')

    def test_draw_three(self):
    	a = Arrows(10)
    	a.draw(2, 5)
    	a.draw(8, 5)
    	a.draw(5, 5)
        self.assertEqual(a.line1.value, r'   \_|_/  ')
        self.assertEqual(a.line2.value, r'     |    ')

class CanvasTest(unittest.TestCase):

	def test_initialize_as_empty(self):
		c = Canvas()
		self.assertEquals(str(c), '')

	def test_append_to_bottom_from_empty(self):
		c = Canvas()
		c.append_to_bottom('abc')
		self.assertEquals(str(c), "abc\n")

	def test_append_to_bottom_with_smaller(self):
		c = Canvas()
		c.append_to_bottom('abc')
		c.append_to_bottom('a')
		self.assertEquals(str(c), "abc\na  \n")

	def test_append_to_bottom_with_larger(self):
		c = Canvas()
		c.append_to_bottom('abc')
		c.append_to_bottom('defghi')
		self.assertEquals(str(c), "abc   \ndefghi\n")

	def test_append_to_bottom_with_same_size(self):
		c = Canvas()
		c.append_to_bottom('abc')
		c.append_to_bottom('def')
		self.assertEquals(str(c), "abc\ndef\n")

	def test_append_to_right_from_empty(self):
		c = Canvas()
		c.append_to_right(self.fill((2, 2), '*'))
		self.assertEquals(str(c), "**\n**\n")

	def test_append_to_right_with_smaller(self):
		c = self.fill((3, 3), '#')
		c.append_to_right(self.fill((2, 2), '*'))
		self.assertEquals(str(c), "###  \n###**\n###**\n")

	def test_append_to_right_with_bigger(self):
		c = self.fill((2, 2), '#')
		c.append_to_right(self.fill((3, 3), '*'))
		self.assertEquals(str(c), "  ***\n##***\n##***\n")

	def fill(self, shape, char):
		c = Canvas()
		for i in xrange(shape[0]):
			c.append_to_bottom(char * shape[1])
		return c

if __name__ == '__main__':
    unittest.main()
