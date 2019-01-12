import i3d
import tensorflow as tf

class I3DInferrer(object):

    def __init__(self, rgb_checkpoint, flow_checkpoint):
        self.rgb_checkpoint = rgb_checkpoint
        self.flow_checkpoint = flow_checkpoint

        self.init_model()

    def init_model(self):
        _IMAGE_SIZE = 224
        _SAMPLE_VIDEO_FRAMES = 79
        NUM_CLASSES = 101
        classes = open('classes.txt').read().splitlines()

        #RGB SETUP
        self.rgb_input = tf.placeholder(
            tf.float32,
            shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
        with tf.variable_scope('RGB'):
            rgb_model = i3d.InceptionI3d(
                NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
            self.rgb_logits, _ = rgb_model(
                self.rgb_input, is_training=False, dropout_keep_prob=1.0)

        rgb_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'RGB':
                rgb_variable_map[variable.name.replace(':0', '').replace('Conv3d',
                                                                         'Conv2d').replace('conv_3d/w',
                                                                                           'weights').replace(
                    'conv_3d/b',
                    'biases').replace('RGB/inception_i3d', 'InceptionV1').replace('batch_norm',
                                                                                  'BatchNorm')] = variable
        self.rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)


        #FLOW SETUP
        self.flow_input = tf.placeholder(
            tf.float32,
            shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2))
        with tf.variable_scope('Flow'):
            flow_model = i3d.InceptionI3d(
                NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
            self.flow_logits, _ = flow_model(
                self.flow_input, is_training=False, dropout_keep_prob=1.0)
        flow_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'Flow':
                flow_variable_map[variable.name.replace(':0', '').replace('Conv3d',
                                                                          'Conv2d').replace('conv_3d/w',
                                                                                            'weights').replace(
                    'conv_3d/b',
                    'biases').replace('Flow/inception_i3d', 'InceptionV1').replace('batch_norm',
                                                                                   'BatchNorm')] = variable
        self.flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

        self.rgb_predictions = tf.nn.softmax(self.rgb_logits)
        self.flow_predictions = tf.nn.softmax(self.flow_logits)

    def infer(self, rgb, flow):
        with tf.Session() as sess:
            feed_dict = {}

            self.rgb_saver.restore(sess, self.rgb_checkpoint)
            feed_dict[self.rgb_input] = rgb
            self.flow_saver.restore(sess, self.flow_checkpoint)
            feed_dict[self.flow_input] = flow

            flow_logit, flow_preds, rgb_logit, rgb_preds = sess.run(
                [self.flow_logits, self.flow_predictions, self.rgb_logits, self.rgb_predictions],
                feed_dict=feed_dict)

            return flow_logit, flow_preds, rgb_logit, rgb_preds