import tensorflow as tf
from tensorflow.keras.models import load_model

model_1 = load_model('./models/model_1.h5')				# 7 days
model_2 = load_model('./models/model_2.h5')				# 30 days
model_3 = load_model('./models/model_3.h5')				# 30 days
model_4 = load_model('./models/model_4.h5')				# 30 days
model_6 = load_model('./models/model_6.h5')				# 8 days
model_9 = load_model('./models/model_9.h5')				# 7 days
turkey_model = load_model('./models/turkey_model.h5')	# 7 days

ensemble_models = [model_1, model_9, turkey_model]

def make_ensemble_preds(data):
	ensemble_preds = []
	for model in ensemble_models:
		preds = model.predict(data)  # make predictions with current ensemble model
		ensemble_preds.append(preds)
	ensemble_preds = tf.squeeze(ensemble_preds)
	ensemble_mean = tf.reduce_mean(ensemble_preds)
	ensemble_mean_float = float(tf.cast(ensemble_mean, tf.float32))

	return ensemble_mean_float
