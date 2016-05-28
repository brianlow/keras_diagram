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
        layers = list(_flatten([node.inbound_layers for node in self.layer.inbound_nodes]))
        layers = [l.layers[-1] if issubclass(type(l), Model) else l for l in layers]
        return [Node(l) for l in layers]

    def _calculate_family_width(self):
        children_width = sum([child.family_width for child in self.children])
        return max([children_width, self.node_width])

    def compress(self):
        self.trim(self.min_text_width())

    def trim(self, text_width):
        for child in self.children:
            child.trim(text_width)
        to_remove = (len(self.text) - text_width) / 2
        if to_remove > 0:
            self.text = self.text[to_remove:-to_remove]
            self.node_width = len(self.text)
            self.family_width = self._calculate_family_width()

    def min_text_width(self):
        t = self.text
        while (t.startswith('  ') and t.endswith('  ')):
            t = t[1:-1]
        return max([len(t)] + [child.min_text_width() for child in self.children])

    def canvas(self):
        canvas = Canvas()
        arrows = []
        offset = 0
        for child in self.children:
            c = child.canvas()
            canvas.append_to_right(c)
            arrows.append((offset + _center_of(child.family_width), _center_of(self.family_width)))
            offset += c.width()
        if len(arrows) > 1:
            a = Arrows(canvas.width())
            for arrow in reversed(sorted(arrows, key=lambda arr: abs(arr[0] - arr[1]))):
                a.draw(arrow[0], arrow[1])
            canvas.append_to_bottom(a.line1.value)
            canvas.append_to_bottom(a.line2.value)
        canvas.append_to_bottom(self.text.center(self.family_width))
        return canvas

    def render(self):
        self.compress()
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
            self.line2[x2] = '|'
            length = (x2) - (x1+2) + 1
            self.line1[x1+2:x2+1] = '_' * length
        elif x1 > x2:
            self.line1[x1-1] = '/'
            self.line2[x2] = '|'
            length = (x1-2) - (x2) + 1
            self.line1[x2:x1-2+1] = '_' * length

def _center_of(width):
    return int((width - 0.5)/2)

def _flatten(items):
    for i in items:
        if hasattr(i, '__iter__'):
            for m in _flatten(i):
                yield m
        else:
            yield i


def ascii(model):
    node = Node(model.layers[-1])
    return node.render()

