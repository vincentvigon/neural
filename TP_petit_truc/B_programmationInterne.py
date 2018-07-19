import tensorflow as tf
import numpy as np

""" on somme les 100 premier carré avec une "dynamic loop"  """
def step0():

    def condition(input, counter):
        return tf.less(counter, 100)


    def loop_body(input, counter):
        output = tf.add(input, tf.square(counter))

        return output, tf.add(counter, 1)


    with tf.name_scope("compute_sum_of_squares"):
        counter = tf.constant(1)
        sum_of_squares = tf.constant(0)

        result = tf.while_loop(condition, loop_body, [sum_of_squares, counter])


    with tf.Session() as sess:
        print(sess.run(result))


""" Encore plus simple. 
Question :  "condition" est testé avant ou après l'appel de "loop_body" ?
En d'autre terme : c'est une boucle 'while' ou bien 'for..while'
"""
def step1():


    def condition(input):
        return tf.less_equal(input, 7)

    def loop_body(input):
        output = input+1
        return output

    with tf.name_scope("compute_sum_of_squares"):
        input = tf.constant(1)
        result = tf.while_loop(condition, loop_body, [input])

    with tf.Session() as sess:
        print(sess.run(result))




















