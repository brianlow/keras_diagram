import unittest
from difflib import ndiff
from diagram import *
from keras.models import *
from keras.layers.core import *
from keras.layers.embeddings import *
from keras.layers.convolutional import *

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
		expected = ("          InputLayer (None, 50, 300)     " + "\n"
					"             Reshape (None, 1, 50, 300)  " + "\n"
					"       Convolution2D (None, 250, 48, 1)  " + "\n"
					"                Relu (None, 250, 48, 1)  " + "\n"
					"        MaxPooling2D (None, 250, 1, 1)   " + "\n"
					"             Flatten (None, 250)         " + "\n"
					"             Dropout (None, 250)         " + "\n"
					"               Dense (None, 7)           " + "\n"
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
		expected = ("          InputLayer (None, 1, 1)                  InputLayer (None, 1, 1)        " + "\n"
					"             Reshape (None, 1, 1, 1)                  Reshape (None, 1, 1, 1)     " + "\n"
					"                    | ______________________________________/                     " + "\n"
					"                    |/                                                            " + "\n"
					"               Merge (None, 2, 1, 1)                                              " + "\n"
				   )
		actual = ascii(model)
		self.assertStringsEqual(actual, expected)

	# TODO: missing uptick on middle stack
	def test_simple_fan_in_three(self):
		one = Sequential()
		one.add(Reshape((1, 1, 1), input_shape=(1,1)))
		two = Sequential()
		two.add(Reshape((1, 1, 1), input_shape=(1,1)))
		three = Sequential()
		three.add(Reshape((1, 1, 1), input_shape=(1,1)))
		model = Sequential()
		model.add(Merge([one, two, three], mode='concat', concat_axis=1))
		expected = ("          InputLayer (None, 1, 1)                  InputLayer (None, 1, 1)                  InputLayer (None, 1, 1)        " + "\n"
					"             Reshape (None, 1, 1, 1)                  Reshape (None, 1, 1, 1)                  Reshape (None, 1, 1, 1)     " + "\n"
					"                    | ______________________________________/________________________________________/                     " + "\n"
					"                    |/                                                                                                     " + "\n"
					"               Merge (None, 3, 1, 1)                                                                                       " + "\n"
				   )
		actual = ascii(model)
		self.assertStringsEqual(actual, expected)

	def test_different_widths_different_heights(self):
		a1 = Sequential()
		a1.add(Reshape((1, 1), input_shape=(1,1)))
		a2 = Sequential()
		a2.add(Reshape((1, 1), input_shape=(1,1)))
		a = Sequential()
		a.add(Merge([a1, a1], mode='concat', concat_axis=1))
		b = Sequential()
		b.add(Reshape((1, 1), input_shape=(1,1)))
		model = Sequential()
		model.add(Merge([a, b], mode='concat', concat_axis=1))
		expected = ("          InputLayer (None, 1, 1)                  InputLayer (None, 1, 1)                                                 " + "\n"
					"             Reshape (None, 1, 1)                     Reshape (None, 1, 1)                                                 " + "\n"
					"                    | ______________________________________/                                                              " + "\n"
					"                    |/                                                                      InputLayer (None, 1, 1)        " + "\n"
					"               Merge (None, 2, 1)                                                              Reshape (None, 1, 1)        " + "\n"
					"                    | _______________________________________________________________________________/                     " + "\n"
					"                    |/                                                                                                     " + "\n"
					"               Merge (None, 3, 1)                                                                                          " + "\n"
				   )
		actual = ascii(model)
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
        self.assertEqual(a.line1.value, r'    \_    ')
        self.assertEqual(a.line2.value, r'      \   ')

    def test_draw_right(self):
    	a = Arrows(10)
    	a.draw(7, 3)
        self.assertEqual(a.line1.value, r'     _/   ')
        self.assertEqual(a.line2.value, r'    /     ')

    def test_draw_multiple_right(self):
    	a = Arrows(10)
    	a.draw(9, 0)
    	a.draw(6, 0)
        self.assertEqual(a.line1.value, r'  ___/__/ ')
        self.assertEqual(a.line2.value, r' /        ')

    def test_draw_both(self):
    	a = Arrows(10)
    	a.draw(3, 3)
    	a.draw(7, 3)
    	a.draw(7, 7)
        self.assertEqual(a.line1.value, r'   | _/|  ')
        self.assertEqual(a.line2.value, r'   |/  |  ')

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
