import tensorflow as tf
import glob
import os
SUB_DIR = ["mask","target"]
class DataGenerator(object):
    def __init__(self,im_dir,batchsize,im_size):
        self.im_dir = im_dir
        self.__loop__()
        self.batchsize   = batchsize
        self.im_size = im_size



    def __call__(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.mask_im,self.target_im))
        dataset = dataset.map(self.process)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(self.batchsize)
        dataset = dataset.prefetch(buffer_size=1000)
        self.dataset = dataset


    def process(self,m_impath,t_impath):
        if m_impath.endswith("png") and t_impath.endswith("png"):
            m_im = tf.image.decode_png(m_impath)
            t_im = tf.image.decode_png(t_impath)
        else:
            m_im = tf.image.decode_jpeg(m_impath)
            t_im = tf.image.decode_jpeg(t_impath)

        m_im = tf.image.resize_images(m_im,self.im_size)
        t_im = tf.image.resize_images(t_im,self.im_size)
        return m_im,t_im

    def __loop__(self):
        self.mask_im = []
        self.target_im = []
        mask_dir = self.im_dir + SUB_DIR[0]
        target_dir = self.im_dir + SUB_DIR[1]
        for file in os.listdir(mask_dir):
            mask_im = mask_dir+"./"+file
            target_im = target_dir+"./"+file
            if os.path.exists(target_im):
                self.mask_im.append(mask_im)
                self.target_im.append(target_im)

        self.datasize =len(self.mask_im)

