import tensorflow as tf
from rescan import *
from data_utils import *
from collections import namedtuple
import os
from ssim import *
import cv2

cfg = namedtuple(cfg,["im_dir","channel","stage_num","use_se","uint","frame","aug_data","batch_size",\
                      "im_size","lr","show_dir","model_dir","depth","decay_step","decay_rate","epoch"])

def path_check(path):
    if not os.path.exists(path):
        os.mkdir(path)

def run(mode,cfg):
    path_check(cfg.model_dir)

    O = tf.placeholder(tf.float32,[cfg.batchsize,cfg.im_size,cfg.im_size,3],name="original")
    B = tf.placeholder(tf.float32,[cfg.batchsize,cfg.im_size,cfg.im_size,3],name="background")
    rain_streaks = DetailNet(input,cfg.channel,kernel_size=[3,3],dilations=[1,1],ratio=6,depth=cfg.depth,\
                             stage_num=cfg.stage_num,frame=cfg.frame,choice=cfg.uint)
    R = O-B
    # reconstruction image
    res_im = 0 - rain_streaks[-1]

    mse_func = lambda i:tf.losses.means_squared_error(label=R,predictions=i)
    ssim_func = lambda j:ssim_block(im1=O-j,im2=O-B,window_size=11,sigma=1.5,size_average=True,channel=3)
    with tf.variable_scope("mse_loss"):
        mse_loss = [mse_func(i) for i in rain_streaks]
        mse_loss = tf.reduce_sum(mse_loss)

    with tf.name_scope("mse_loss"):
        mse_sum = tf.summary.scalar("mse_loss",mse_loss)

    with tf.variable_scope("ssim_func"):
        ssim_loss = [ssim_func(j) for j in rain_streaks]

    ssim_sum = list()
    with tf.name_scope("ssim_loss"):
        for idx,item in enumerate(ssim_loss):
            ssim_loss_name = "ssim_loss_"+str(i)
            ssim_sum.append(tf.summary.scalar(ssim_loss_name,item))

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(learning_rate=cfg.lr,global_step=global_step,decay_steps=cfg.decay_step,decay_rate=cfg.decay_rate,staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-8)

    update_vars = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_ops = tf.group(*update_vars)

    with tf.control_dependencies([update_ops]):
        train_op  = optimizer.minimizer(mse_loss,global_step)

    summary = tf.summary.merge_all()
    s_writer = tf.summary.FileWriter(cfg.model_dir)

    if mode == "train":
        train_data = DataGenerator(cfg.im_dir,cfg.batchsize,cfg.im_size)
        train_iterator = tf.data.Iterator.from_structure(train_data.data.output_types,train_data.data.output_shapes)
        train_next_batch = train_iterator.get_next()
        train_init       = train_iterator.make_initializer(train_data.data)
        train_iterations = train_data.datasize/cfg.batchsize

        max_steps        = train_iterations*cfg.epoch
        run_step = 0

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            s_writer.add_graph(sess.graph)
            for i in range(cfg.epoch):
                sess.run(train_init)
                for j in range(train_iterations):
                    mask_batch,target_batch = sess.run(train_next_batch)
                    _,m_loss,s_loss,b_res_im,b_sum = sess.run([train_op,mse_loss,ssim_loss,res_im,summary],feed_dict={O:mask_batch,B:target_batch})
                    run_step += 1
                    if (run_step%200==0):
                        s_writer.add_summary(b_sum,run_step)
                    if (run_step%1000==0):
                        mask_im = np.squeeze(mask_batch[1,:,:,:])
                        target_im = np.squeeze(target_batch[1,:,:,:])
                        result_im = np.squeeze(b_res_im[1,:,:,:])
                        concate_im = np.hstack([mask_im,target_im,result_im])
                        save_name = cfg.show_dir + "epoch_"+str(i)+"iteraction_"+str(j)+".jpg"
                        cv2.imwrite(save_name,concate_im)

    elif mode=="test":
        test_data = DataGenerator(cfg.im_dir,cfg.batchsize,cfg.im_size)
        test_iterator = tf.data.Iterator.from_structure(test_data.data.output_types,test_data.data.output_shapes)
        test_next_batch = test_iterator.get_next()
        test_init       = test_iterator.make_initializer(test_data.data)
        test_iterations = test_data.datasize/cfg.batchsize

        latest_checkpoint = tf.train.latest_checkpoint(cfg.model_dir)

        with tf.Session() as sess:
            sess.restore(sess,latest_checkpoint)
            sess.run(test_init)
            for i in range(test_iterations):
                mask_batch,target_batch = sess.run(test_next_batch)
                b_res_im = sess.run([res_im],feed_dict={O:mask_batch,B:target_batch})
                for i in range(mask_batch.shape[0]):
                    mask_im = np.squeeze(mask_batch[i,:,:,:])
                    target_im = np.squeeze(target_batch[i,:,:,:])
                    result_im  = np.squeeze(b_res_im[i,:,:,:])
                    concat_im = cv2.hstack([mask_im,result_im,target_im])
                    save_name = cfg.show_dir+str(i)+".jpg"
                    cv2.imwrite(save_name,concat_im)
    else:
        print("export mode does not implement")



