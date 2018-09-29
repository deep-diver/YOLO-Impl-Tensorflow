import tensorflow as tf
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import avg_pool2d
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected

def create_model():
    input = tf.placeholder(tf.float32, [None, 448, 448, 3], name='input')
    output = tf.placeholder(tf.float32, [None, 7, 7, 30], name='output')

    conv1 = conv2d(input, num_outputs=64,
                  kernel_size=[7,7], stride=2, padding='SAME',
                  activation_fn=tf.nn.relu)
    conv1 = max_pool2d(conv1, kernel_size=[2,2], stride=2, padding='VALID')

    conv2 = conv2d(conv1, num_outputs=192,
                  kernel_size=[3,3], stride=1, padding='SAME',
                  activation_fn=tf.nn.relu)
    conv2 = max_pool2d(conv2, kernel_size=[2,2], stride=2, padding='VALID')

    conv3 = conv2d(conv2, num_outputs=128,
                  kernel_size=[1,1], stride=1, padding='SAME',
                  activation_fn=tf.nn.relu)
    conv3 = conv2d(conv3, num_outputs=256,
                  kernel_size=[3,3], stride=1, padding='SAME',
                  activation_fn=tf.nn.relu)
    conv3 = conv2d(conv3, num_outputs=256,
                  kernel_size=[1,1], stride=1, padding='SAME',
                  activation_fn=tf.nn.relu)
    conv3 = conv2d(conv3, num_outputs=512,
                  kernel_size=[3,3], stride=1, padding='SAME',
                  activation_fn=tf.nn.relu)
    conv3 = max_pool2d(conv3, kernel_size=[2,2], stride=2, padding='VALID')
    prev = conv3

    for i in range(4):
        conv4 = conv2d(prev, num_outputs=256,
                      kernel_size=[1,1], stride=1, padding='SAME',
                      activation_fn=tf.nn.relu)
        conv4 = conv2d(conv4, num_outputs=512,
                      kernel_size=[3,3], stride=1, padding='SAME',
                      activation_fn=tf.nn.relu)
        prev = conv4

    conv4 = conv2d(prev, num_outputs=512,
                  kernel_size=[1,1], stride=1, padding='SAME',
                  activation_fn=tf.nn.relu)
    conv4 = conv2d(prev, num_outputs=1024,
                  kernel_size=[3,3], stride=1, padding='SAME',
                  activation_fn=tf.nn.relu)
    conv4 = max_pool2d(conv4, kernel_size=[2,2], stride=2, padding='VALID')
    prev = conv4

    for i in range(2):
        conv5 = conv2d(prev, num_outputs=512,
                      kernel_size=[1,1], stride=1, padding='SAME',
                      activation_fn=tf.nn.relu)
        conv5 = conv2d(conv5, num_outputs=1024,
                      kernel_size=[3,3], stride=1, padding='SAME',
                      activation_fn=tf.nn.relu)
        prev = conv5

    conv5 = conv2d(prev, num_outputs=1024,
                  kernel_size=[3,3], stride=1, padding='SAME',
                  activation_fn=tf.nn.relu)
    # ?
    conv5 = conv2d(conv5, num_outputs=1024,
                  kernel_size=[3,3], stride=2, padding='SAME',
                  activation_fn=tf.nn.relu)

def main():
    printf('hello world')

if __name__ == "__main__":
    main()
