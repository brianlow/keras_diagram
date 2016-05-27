# todo - customized format string (name, classname, shape, shape_without_none, shape_bare) - maybe accept func
# todo - remove whitespace border (better is remove whitespace per column, good enough may be customize format string)
# todo - width of each branch dependent on text in the branch
# todo - collapse whitespace on either side of layer list (use max width of layers in a node)
# todo - colored output
# todo - better name?

import numpy as np
import string
import ctypes
from ctypes import create_string_buffer
from keras.models import Model
from keras.layers.core import *
import pdb

class Node:
    def __init__(self, layer):
        self.layer = layer
        self.text = "%20s %-20s" %  (self._name(), layer.output_shape)
        self.children = self._calculate_children()
        self.node_width = len(self.text)
        self.family_width = self._calculate_family_width()
        self.present = True

    def _name(self):
        if type(self.layer) is Activation:
            return self.layer.activation.func_name.title()
        else:
            return self.layer.__class__.__name__

    def _calculate_children(self):
        layers = list(self._flatten([node.inbound_layers for node in self.layer.inbound_nodes]))
        layers = [l.layers[-1] if issubclass(type(l), Model) else l for l in layers]
        return [Node(l) for l in layers]

    def _calculate_family_width(self):
        children_width = sum([child.family_width for child in self.children])
        return max([children_width, self.node_width])

    def _flatten(self, items):
        for i in items:
            if hasattr(i, '__iter__'):
                for m in self._flatten(i):
                    yield m
            else:
                yield i

    def canvas(self):
        canvas = Canvas()
        arrows = []
        offset = 0
        for child in self.children:
            c = child.canvas()
            canvas.append_to_right(c)
            arrows.append((offset + (self.node_width/2), self.node_width/2))
            offset += c.width()
        if len(arrows) > 1:
            a = Arrows(canvas.width())
            for arrow in reversed(arrows):
                a.draw(arrow[0], arrow[1])
            canvas.append_to_bottom(a.line1.value)
            canvas.append_to_bottom(a.line2.value)
        canvas.append_to_bottom(self.text)
        return canvas

    def render(self):
        return str(self.canvas())

class Canvas:
    def __init__(self):
        self.chars = self._empty((0,0))

    def __str__(self):
        s = ""
        for i in xrange(len(self.chars)):
            s += string.join(self.chars[i], '') + "\n"
        return s

    def height(self):
        return self.chars.shape[0]

    def width(self):
        return self.chars.shape[1]

    def append_to_bottom(self, text):
        new_height = self.height() + 1
        new_width = max(self.width(), len(text))
        self._expand((new_height, new_width))
        self.chars[-1] = list(text.ljust(new_width))

    def append_to_right(self, canvas):
        x1 = self.height()
        x2 = canvas.height()
        new_height = max(x1, x2)
        #new_height = max(self.height(), canvas.height())
        new_width = self.width() + canvas.width()
        self._expand((new_height, new_width), down=False)
        self.chars[(self.height()-canvas.height()):,self.width()-canvas.width():] = canvas.chars

    def _expand(self, shape, right=True, down=True):
        if self.chars.shape == (0, 0):
            self.chars = self._empty(shape)
            return
        # expand height (add rows)
        rows_to_add = shape[0] - self.height();
        if rows_to_add > 0:
            new_rows = self._empty((rows_to_add, self.width()))
            if down:
                self.chars = np.append(self.chars, new_rows, axis=0)
            else:
                self.chars = np.append(new_rows, self.chars, axis=0)
        # expand width (add columns)
        cols_to_add = shape[1] - self.width();
        if cols_to_add > 0:
            new_cols = self._empty((self.height(), cols_to_add))
            if right:
                self.chars = np.append(self.chars, new_cols, axis=1)
            else:
                self.chars = np.append(new_cols, self.chars, axis=1)

    def _empty(self, shape):
        arr = np.empty(shape, dtype=str)
        arr[:,:] = ' '
        return arr


class Arrows:
    def __init__(self, width):
        self.width = width
        self.arrows = []
        self.line1 = create_string_buffer(' ' * width)
        self.line2 = create_string_buffer(' ' * width)

    def any(self):
        return self.line1.value.strip() and self.line2.value.strip()

    def draw(self, x1, x2):
        if x1 == x2:
            self.line1[x1] = '|'
            self.line2[x2] = '|'
        elif x1 < x2:
            self.line1[x1+1] = '\\'
            self.line2[x2-1] = '\\'
            length = (x2-2) - (x1+2) + 1
            self.line1[x1+2:x2-2+1] = '_' * length
        elif x1 > x2:
            self.line1[x1-1] = '/'
            self.line2[x2+1] = '/'
            length = (x1-2) - (x2+2) + 1
            self.line1[x2+2:x1-2+1] = '_' * length


def ascii(model):
    node = Node(model.layers[-1])
    return node.render()

