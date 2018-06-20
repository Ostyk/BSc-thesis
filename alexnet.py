import tensorflow as tf
import numpy as np

class Alexnet(object):
    def __init__(self,x,weights_path,transfer_learning=True):

        self.X = x
        self.weights_path = weights_path
        self.transfer_learning = transfer_learning

        #x = tf.placeholder(tf.float32, (None,) + xdim)
        self.network()

    def load_weights(self, name):
        '''loads weights and returns weights and biases of a given layer'''

        net_data = np.load(self.weights_path, encoding='bytes').item()
        return tf.Variable(net_data[name][0]), tf.Variable(net_data[name][1])

    def maxpool(self,input,k_h,k_w,s_h,s_w,padding = 'VALID'):
        '''
        k - spatial extent
        h - height
        w - width
        s - strides
        '''
        return tf.nn.max_pool(input, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    def avgpool(self,input,k_h,k_w,s_h,s_w,padding = 'VALID'):
        '''
        k - spatial extent
        h - height
        w - width
        s - strides
        '''
        return tf.nn.avg_pool(input, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    def fc(self,input,name):
        '''first conv'''

        kernel, biases = self.load_weights(name=name)

        return tf.nn.relu_layer(tf.reshape(input, [1, int(np.prod(input.get_shape()[1:]))]), kernel, biases)

    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, padding="SAME", group=1):
        '''
        k - spatial extent
        h - height
        w - width
        s - strides
        '''
        kernel, biases = self.load_weights(name=name)

        c_i = input.get_shape()[-1]
        assert c_i%group==0
        assert c_o%group==0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

        if group==1:
            conv = convolve(input, kernel)
        else:
            input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
            kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
            conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
        return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

    def lrn(self,input, radius, alpha, beta, bias=1.0):
        ''' normalization '''
        return tf.nn.local_response_normalization(input,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    def dropout(self,input):
        '''dropout'''
        return tf.nn.dropout(input)

    def relu(self,input):
        '''Rectified linear units activation function'''
        return tf.nn.relu(input)

    def network(self,mx5_params=None):
        '''
        network structure
        '''
        #1st convolutional layerxw
        conv1 = self.conv(self.X, 11, 11, 96, 4, 4, name='conv1',group=1,padding="SAME")
        relu1 = self.relu(conv1)
        lrn1 = self.lrn(relu1, radius = 2, alpha = 2e-05, beta = 0.75, bias = 1.0)
        maxpool1 = self.maxpool(lrn1, k_h = 3, k_w = 3, s_h = 2, s_w = 2, padding = 'VALID')

        #2nd convolutional layer
        conv2 = self.conv(maxpool1, 5, 5, 256, 1, 1, group = 2, name='conv2',padding="SAME")
        relu2 = self.relu(conv2)
        lrn2 = self.lrn(relu2, radius = 2, alpha = 2e-05, beta = 0.75, bias = 1.0)
        maxpool2 = self.maxpool(lrn2, k_h = 3, k_w = 3, s_h = 2, s_w = 2, padding = 'VALID')

        #3rd convolutional layer
        conv3 = self.conv(maxpool2, 3, 3, 384, 1, 1, group = 1, name='conv3')
        relu3 = self.relu(conv3)

        #4th convolutional layer
        conv4 = self.conv(relu3, 3, 3, 384, 1, 1, group = 2, name='conv4')
        relu4 = self.relu(conv4)

        #5th convolutional layer
        conv5 = self.conv(relu4, 3, 3, 256, 1, 1, group = 2, name='conv5')
        #self.conv5 = self.conv(relu4, 3, 3, 256, 1, 1, group = 2, name='conv5')
        relu5 = self.relu(conv5)
        if mx5_params:
            k_h, k_w, s_h, s_w, pooling = mx5_params
            if pooling == 'maxpool':
                return self.maxpool(relu5, k_h, k_w, s_h, s_w, padding = 'VALID')
            elif pooling == 'avgpool':
                return self.avgpool(relu5, k_h, k_w, s_h, s_w, padding = 'VALID')
        else:
            maxpool5 = self.maxpool(relu5, k_h = 3, k_w = 3, s_h = 2, s_w = 2, padding = 'VALID')
            # 6th layerx
            fc6 = self.fc(maxpool5,name='fc6')
            return fc6
            #dropout
            #fc7 = tf.nn.relu_layer(fc6, kernel, biases#dropout
            #fc8 = tf.nn.xw_plus_b(fc7, kernel, biases) #dropout
            #prob = tf.nn.softmax(fc8)


        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        ##flattened = tf.reshape(pool5, [-1, 6*6*256])
        ##fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
        #dropout6 = dropout(fc6, self.KEEP_PROB)

        # 7th Layer: FC (w ReLu) -> Dropout
        #fc7 = fc(dropout6, 4096, 4096, name='fc7')
        #dropout7 = dropout(fc7, self.KEEP_PROB)

        # 8th Layer: FC and return unscaled activations
        #self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')

        #return maxpool5
        #print("F")
        #if self.transfer_learning:
        #    self.maxpool5 = self.maxpool(lrn2,k_h = 3, k_w = 3, s_h = 2, s_w = 2, padding = 'VALID')
