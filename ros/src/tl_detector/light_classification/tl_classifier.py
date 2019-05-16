from styx_msgs.msg import TrafficLight
import numpy as np
import tensorflow as tf
import datetime

#PATH_TO_MODEL = '~/Desktop/CarND-Capstone-master/ros/src/tl_detector/light_classification/capstone/out_frozen_sim/frozen_inference_graph.pb'
class TLClassifier(object):
    def __init__(self):
        PATH_TO_MODEL = r'light_classification/frozen_inference_graph.pb'
        self.threshold = .5
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)
    
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(image, axis=0)  
            start = datetime.datetime.now()
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
            end = datetime.datetime.now()
            c = end - start
            print('TIME(S) : 'c.total_seconds())

        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        print('SCORES  : ', scores[0])
        print('CLASSES : ', classes[0])

        if scores[0] > self.threshold:
            if classes[0] == 1:
                print('STATE : GREEN')
                return TrafficLight.GREEN
            elif classes[0] == 2:
                print('STATE : RED')
                return TrafficLight.RED
            elif classes[0] == 3:
                print('STATE : YELLOW')
                return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN
        #return boxes, scores, classes, num
        #return TrafficLight.UNKNOWN
