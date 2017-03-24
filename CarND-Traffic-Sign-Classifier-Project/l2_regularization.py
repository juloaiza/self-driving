# Original loss function (ex: classification using cross entropy)
unregularized_loss = tf.nn.sigmoid_cross_entropy_with_logits(predictions, labels) 
 
# Regularization term, take the L2 loss of each of the weight tensors, 
# in this example, 2 convolutional layers and a fully connected layer. 
# Sum them and multiply by a hyper-parameter controlling the amount of L2 loss
l2_loss = l2_regularization_penalty * (tf.nn.l2_loss(W_conv1) + 
                                       tf.nn.l2_loss(W_conv2) +
                                       tf.nn.l2_loss(W_fc1)) 
 
# Simply add L2 loss to your unregularized loss 
loss = tf.add(unregularized_loss, l2_loss, name='loss')