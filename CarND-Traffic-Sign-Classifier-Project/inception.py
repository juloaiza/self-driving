### Define your architecture here.
### Feel free to use as many code cells as needed.
from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0.0001#0.0001
    sigma = 0.01#0.01
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W1 = tf.Variable(tf.truncated_normal(shape=(3, 3, 3, 32), mean = mu, stddev = sigma))
    conv1_b1 = tf.Variable(tf.zeros(32))
    conv11   = tf.nn.conv2d(x, conv1_W1, strides=[1, 1, 1, 1], padding='SAME') + conv1_b1
    
    # TODO: Activation.
    #conv11 = tf.nn.relu(conv11)
    #print(tf.shape(conv11))

    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 32), mean = mu, stddev = sigma))
    conv1_b2 = tf.Variable(tf.zeros(32))
    conv12   = tf.nn.conv2d(x, conv1_W2, strides=[1, 1, 1, 1], padding='SAME') + conv1_b2
    
    # TODO: Activation.
    #conv12 = tf.nn.relu(conv12)
    #print(tf.shape(conv12))
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W3 = tf.Variable(tf.truncated_normal(shape=(1, 1, 3, 32), mean = mu, stddev = sigma))
    conv1_b3 = tf.Variable(tf.zeros(32))
    conv13   = tf.nn.conv2d(x, conv1_W3, strides=[1, 1, 1, 1], padding='SAME') + conv1_b3
    
    conv1 = tf.nn.relu(tf.concat([conv11, conv12, conv13],2))
    
    
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6. New I=30x30x32 O=15x15x32
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO: Layer 2: Convolutional. Output = 10x10x16. O=13x13x64
    conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 64), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(64))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
        
    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)
    
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16. New I=13x13x64 O=6x6x64
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')




    # TODO: Layer 3: Convolutional. Output = 10x10x16. O=13x13x64
    conv3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 128), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(128))
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b
        
    # TODO: Activation.
    conv3 = tf.nn.relu(conv3)
    
   # TODO: Pooling. Input = 10x10x16. Output = 5x5x16. New I=13x13x21, 1, 1], padding='VALID')
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
         
   

    # add dropout on hidden layer
    
    #conv3 = tf.nn.dropout(conv3, 0.4)    
    
    
    
    
    # TODO: Flatten. Input = 5x5x16. Output = 400 (5x5x16).
    #5x5x64
    fc0   = flatten(conv3)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(2560, 400), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(400))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # TODO: Activation.
    fc1    = tf.nn.relu(fc1)   

    # TODO: Layer 5: Fully Connected. Input = 120. Output = 84.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(120))
    fc3    = tf.matmul(fc1, fc3_W) + fc3_b    
    
    # TODO: Activation.
    fc3    = tf.nn.relu(fc3)     
    
    
    # TODO: Layer 6: Fully Connected. Input = 84. Output = 43.
    fc4_W  = tf.Variable(tf.truncated_normal(shape=(120, 43), mean = mu, stddev = sigma))
    fc4_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc3, fc4_W) + fc4_b    
    
    return logits