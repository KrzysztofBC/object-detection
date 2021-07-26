import tensorflow as tf
from object_detection import tf_hub_sample


def test_loading_model_from_tf_hub():
	tf_hub_det = tf_hub_sample.TfHubSampleDetector()
	output = tf_hub_det.detector(tf.ones((1, 256, 256, 3), tf.uint8))
	assert output["detection_anchor_indices"].numpy().shape == (1, 100)
