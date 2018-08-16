import tensorflow as tf
import numpy as np
import collections
"""
all assumed list value of kernel_size is equal
"""
cfg = collections.namedtuple("cfg",["batchsize","num_filters","stage_num","depth","dilations","kernel_size","pad_size","choice","frame"])


def Conv2d(input,num_filters,ksize=[3,3],dilations=[1,1],strides=[1,1],pad="VALID",name="conv"):
    n,h,w,c = input.get_shape().as_list()
    with tf.variable_scope(name):
        weight = tf.get_variable("weight",ksize+[c,num_filters],dtype=tf.float32,initializer=tf.glorot_uniform_initializer())
        bias   = tf.get_variable("bias",[num_filters],dtype=tf.float32,initializer=tf.zeros_initializer())
        conv   = tf.nn.conv2d(input,weight,[1]+strides+[1],padding=pad,dilations=[1]+dilations+[1])
        conv   = tf.nn.bias_add(conv,bias)
    return conv

def Conv2d_pad(input,num_filters,ksize=[3,3],strides=[1,1],dilations=[1,1],pad_size=0,pad="VALID",name="conv_pad"):
    n,h,w,c = input.get_shape().as_list()
    with tf.variable_scope(name):
        if pad_size>0:
            input = tf.pad(input,[[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]],name="pad_input")
        weight = tf.get_variable("weight",ksize+[c,num_filters],dtype=tf.float32,initializer=tf.glorot_uniform_initializer())
        bias   = tf.get_variable("bias",[num_filters],dtype=tf.float32,initializer=tf.zeros_initializer())
        conv   = tf.nn.conv2d(input,weight,[1]+strides+[1],padding=pad,dilations=[1]+dilations+[1])
        conv   = tf.nn.bias_add(conv,bias)
    return conv



def Snorm(input,name="switch_norm"):
    n,h,w,c = input.get_shape().as_list()
    eps     = 1e-3
    with tf.variable_scope(name):

        weight = tf.get_variable("weight",[1,1,1,c],dtype=tf.float32,initializer=tf.ones_initializer())
        bias   = tf.get_variable("bias",[c],dtype=tf.float32,initializer=tf.zeros_initializer())
        mean_weight = tf.get_variable("weight_mean",[3],dtype=tf.float32,initializer=tf.ones_initializer())
        var_weight  = tf.get_variable("weight_var",[3],dtype=tf.float32,initializer=tf.ones_initializer())

        # in batch
        mean_ins,var_ins = tf.nn.moments(input,axes=[1,2],keep_dims=True)

        mean_ln,var_ln   = tf.nn.moments(input,axes=[1,2,3],keep_dims=True)

        mean_bn,var_bn   = tf.nn.moments(input,axes=[0,1,2],keep_dims=True)

        mean_weight  = tf.nn.softmax(mean_weight)
        var_weight   = tf.nn.softmax(var_weight)

        mean  = mean_weight[0]*mean_ins + mean_weight[1]*mean_ln + mean_weight[2]*mean_bn
        var   = var_weight[0]*var_ins   + var_weight[1]*var_ln   + var_weight[2]*var_bn

        norm = (input-mean)/tf.sqrt(var+eps)
        norm = norm*weight + bias
        """
        offset = tf.get_variable("offset",[c],dtype=tf.float32,initializer=tf.zeros_initializer())
        scale  = tf.get_variable("scale",[c],dtype=tf.float32,initializer=tf.random_normal_initializer(0,0.2))
        mean,variance = tf.nn.moments(input,axes=[0,1,2],keep_dims=False)
        variance_epsilon = 1e-3
        norm = tf.nn.batch_normalization(input,mean,variance,offset,scale,variance_epsilon=variance_epsilon)
        """
    return norm


def Global_Pool(input,name="global_pooling"):
    w,h,w,c  = input.get_shape().as_list()
    with tf.variable_scope(name):
        output = tf.nn.avg_pool(input,[1,h,w,1],[1,1,1,1],padding="VALID")
    return output



def SE(input,num_filters,ratio,name="SE_block"):
    with tf.variable_scope(name):
        output = Global_Pool(input)
        output = Conv2d(output,num_filters/ratio,[1,1],name="fc_1")
        output = tf.nn.relu(output)
        output = Conv2d(output,num_filters,[1,1],name="fc_2")
        output = tf.sigmoid(output,name="sigmoid")
        output = input*output
    return output


def DireConv(input,num_filters,kernel_size,dilations,ratio,pair=None,name="DireConv"):
    with tf.variable_scope(name):
        pad = int(dilations * (kernel_size - 1) / 2)
        input = tf.pad(input,[[0,0],[pad,pad],[pad,pad],[0,0]],name="input")
        conv  = Conv2d(input,num_filters,kernel_size,dilations)
        conv  = SE(conv,num_filters,ratio)
        conv  = tf.nn.leaky_relu(conv,alpha=0.2)
    return conv,None




def RNN(input,num_filters,kernel_size,dilations,pair=None,name="RNN"):
    with tf.variable_scope(name):
        pad_x    = int(dilations[0]*(kernel_size[0]-1)/2)
        input_x  = tf.pad(input,[[0,0],[pad_x,pad_x],[pad_x,pad_x],[0,0]],name="pad_x")
        conv_x   = Conv2d(input_x,num_filters,kernel_size,dilations=dilations)

        pad_h   = int((kernel_size[0])/2)
        input_h = tf.pad(input,[[0,0],[pad_h,pad_h],[pad_h,pad_h],[0,0]],name="pad_h")
        conv_h  = Conv2d(input_h,num_filters,kernel_size,name="conv_h")

        if pair is not None:
            h = tf.tanh(conv_x+conv_h)
        else:
            h = tf.tanh(conv_x)

        h = tf.nn.leaky_relu(h,alpha=0.2)
        return h,h

def LSTM(input,num_filters,kernel_size,dilations,ratio,pair=None,name="LSTM"):
    with tf.variable_scope(name):
        pad_x = int(dilations*(kernel_size-1)/2)
        input_x = tf.pad(input,[[0,0],[pad_x,pad_x],[pad_x,pad_x],[0,0]],name="input_x")
        conv_xf = Conv2d(input_x,num_filters,kernel_size,dilations,name="conv_xf")
        conv_xi = Conv2d(input_x,num_filters,kernel_size,dilations,name="conv_xi")
        conv_xo = Conv2d(input_x,num_filters,kernel_size,dilations,name="conv_xo")
        conv_xj = Conv2d(input_x,num_filters,kernel_size,dilations,name="conv_xj")

        pad_h   = int((kernel_size-1)/2)
        input_h = tf.pad(input,[[0,0],[pad_h,pad_h],[pad_h,pad_h],[0,0]],name="input_y")
        conv_hf = Conv2d(input_h,num_filters,kernel_size,name="conv_hf")
        conv_hi = Conv2d(input_h,num_filters,kernel_size,name="conv_hi")
        conv_ho = Conv2d(input_h,num_filters,kernel_size,name="conv_hi")
        conv_hj = Conv2d(input_h,num_filters,kernel_size,name="conv_hi")

        if pair is not None:
            h,c = pair
            f   = tf.sigmoid(conv_xf+conv_hf)
            i   = tf.sigmoid(conv_xi+conv_hi)
            o   = tf.sigmoid(conv_xo+conv_ho)
            j   = tf.tanh(conv_xj+conv_hj)
            c   = f*c + i*j
            h   = o*c
        else:
            i   = tf.sigmoid(conv_xi)
            o   = tf.sigmoid(conv_xo)
            j   = tf.tanh(conv_xj)
            c   = i*j
            h   = o*c

        output = SE(h,num_filters,ratio)
        output = tf.nn.leaky_relu(output,alpha=0.2)
    return output,[output,c]

def GRU(input,num_filters,kernel_size,dilations,ratio,pair=None,name="GRU"):
    with tf.variable_scope(name):
        pad_x = int(dilations * (kernel_size - 1) / 2)
        input_x = tf.pad(input, [[0, 0], [pad_x, pad_x], [pad_x, pad_x], [0, 0]], name="input_x")
        conv_xz = Conv2d(input_x, num_filters, kernel_size, dilations, name="conv_xz")
        conv_xr = Conv2d(input_x, num_filters, kernel_size, dilations, name="conv_xr")
        conv_xn = Conv2d(input_x, num_filters, kernel_size, dilations, name="conv_xn")

        pad_h = int((kernel_size - 1) / 2)
        input_h = tf.pad(input, [[0, 0], [pad_h, pad_h], [pad_h, pad_h], [0, 0]], name="input_y")
        conv_hz = Conv2d(input_h, num_filters, kernel_size, name="conv_hz")
        conv_hr = Conv2d(input_h, num_filters, kernel_size, name="conv_hr")
        #conv_hn = Conv2d(input_h, num_filters, kernel_size, name="conv_hn")

        if pair is not None:
            z = tf.sigmoid(conv_xz+conv_hz)
            r = tf.sigmoid(conv_xr+conv_hr)
            n = tf.tanh(conv_xn+Conv2d_pad(r*pair,num_filters,kernel_size,pad_size=pad_h,name="r*pair"))
            h = (1-z)*pair +z*n
        else:
            z = tf.sigmoid(conv_xz)
            f = tf.tanh(conv_xn)
            h = z*f

        output = SE(h,num_filters,ratio)
        output = tf.nn.leaky_relu(output,alpha=0.2)
    return output,output




def Basic_rnn_block(input,num_filters,kernel_size,dilations,depth,state,choice="RNN",name="RNN_block"):
    """
    :param state: note state should be list,length equals to nums of rnn_unit
    :return:
    """
    rnn_map = {"GRU":GRU,"RNN":RNN,"LSTM":LSTM,"DireConv":DireConv}
    cnt = 0
    tmp_state = []
    with tf.variable_scope(name):
        conv_name = choice+"_"+str(cnt)
        conv,cur_state = rnn_map[choice](input=input,num_filters=num_filters,kernel_size=kernel_size,dilations=dilations,pair=state[cnt],name=conv_name)
        tmp_state.append(cur_state)
        for i in range(depth-3):
            cnt += 1
            conv_name = choice+"_"+str(cnt)
            conv,cur_state = rnn_map[choice](conv,num_filters=num_filters,kernel_size=kernel_size,dilations=[(2**i)*dilations[0],(2**i)*dilations[1]],pair=state[cnt],name=conv_name)
            tmp_state.append(cur_state)
    return conv,tmp_state,cnt

def Basic_dec_block(input,num_filters,kernel_size,dilations,ratio,name="Dec_block"):
    with tf.variable_scope(name):
        conv = Conv2d_pad(input,num_filters=num_filters,ksize=kernel_size,dilations=dilations,pad_size=1)
        conv = SE(conv,num_filters,ratio)
        conv = tf.nn.leaky_relu(conv,alpha=0.2)
        conv = Conv2d(conv,num_filters=3,ksize=[1,1])
    return conv


def DetailNet(input,num_filters,kernel_size,dilations,ratio,depth,stage_num,frame,choice="RNN",name="DetailNet"):
    # return is rain result
    copy_x = input # may be it should be deepcopy?
    res    = []
    with tf.variable_scope(name):
        init_state = [None for _ in range(1+depth-3)]
        for i in range(stage_num):
            stage_name = "stage_"+str(i)
            conv,c_state,_ = Basic_rnn_block(input,num_filters,kernel_size,dilations,depth,init_state,choice,name=stage_name+'RNN_Block')
            conv           = Basic_dec_block(conv,num_filters,kernel_size,dilations,ratio=ratio,name=stage_name+"Dec_Block")

            if frame == "add" and i>0:
                conv = conv + res[-1]
            res.append(conv)
            init_state = c_state
            input      = copy_x - conv
    return res

if __name__ == "__main__":
    cfg.batchsize = 1
    cfg.num_filters = 32
    cfg.kernel_size = [3,3]
    cfg.dilations   = [1,1]
    cfg.stage_num   = 2
    cfg.choice      = "RNN"
    cfg.depth       = 6
    cfg.ratio       = 6
    cfg.frame       = "add"
    input           = tf.placeholder(tf.float32,[cfg.batchsize,224,224,3],name="input")
    res = DetailNet(input,cfg.num_filters,cfg.kernel_size,cfg.dilations,cfg.ratio,cfg.depth,cfg.stage_num,cfg.frame)
