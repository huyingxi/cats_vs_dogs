import tensorflow as tf
def inference(images, batch_size, n_classes):
    '''
    build model
    args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    return:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                 shape = [3,3,3,16],
                                 dtype = tf.float32,
                                 initializer = tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variables('biases',
                                 shape = [16],
                                 dtype = tf.float32,
                                 initializer = tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name = scope.name)
        
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize = [1,3,3,1], strides = [1,2,2,1],
                              padding = 'SAME', name = 'pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias = 1.0, alpha=0.001/9.0,
                         beta=0.75,name='norm1')
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                 shape=[3,3,16,16],
                                 dtype=tf.float32,
                                 initializer = tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                shape=[16],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1], padding = 'SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha = 0.001/9.0,
                         beta=0.75,name='norm2')
        pool2 = tf.nn.max_pool(norm2,ksize=[1,3,3,1],strides=[1,1,1,1],
                              padding='SAME',name='pooling2')
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2,shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                 shape=[dim,128],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initalizer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                shape=[128],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights)+biases, name=scope.name)
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                 shape=[128.128],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                shape=[128],
                                dtype = tf.float32,
                                initializer = tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3,weights)+biases, name='local4')
    with tf.variable_scope('softmax_layer') as scope:
        weights = tf.get_variable('softmax_linear',
                                 shape=[128,n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.lofat32))
        biases = tf.get_variable('biases',
                                shape=[n_classes],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0.1))
        softmax_liner=tf.add(tf.matmul(local4, weights),biases,name='softmax_linear')
    return softmax_layer


def losses(logits,labels):
    '''
    compute loss from logits and labels
    args:
        logits: logit tensor, float , [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]
    returns:
        loss tensor of float type
    '''
    
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = logits, labels = labels,name='xentropy_re_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss',loss)
    return loss
    
    
def training(loss, learning_rate):
    '''
    training ops, the op returned by this function is what must be passed to 'sess.run()' 
    call to cause the model to train.
    args:
        loss : loss tensor, from losses()
    returns:
        train_op: the op for training
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0,name='global_step',trainable=False)
        train_op = optimizer.minimize(loss, global_step = global_step)
    return train_op
    

def evalutaion(logits, lables):
    '''
    evaluate the quality of the logits at prediction the lable.
    args:
        logits: logits tensor, float, [batch_size, num_classes]
        lables: lables tensor,tf.int32, [batch_size], with values in the 
        range [0,num_classes]
    returns:
        a scalar int32 tensor with the number of examples(out of batch_size)
        that were predicted correctly.
    '''
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels,1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy',accuracy)
    return accuracy

