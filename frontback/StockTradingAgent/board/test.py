from tensorflow import keras
from keras.models import load_model
# from keras.models import load_weights
from keras.models import model_from_json
 
model = keras.models.load_model('model/a3c_dnn_value_20201109025659.h5')
print('loading success')
python main.py --stock_code 005930 --rl_method a2c --net lstm --num_steps 5 --output_name test_005930 --num_epoches 1 --start_epsilon 0 --start_date 20180101 --end_date 20181231 --reuse_models --value_network_name a2c_lstm_policy_005930 --policy_network_name a2c_lstm_value_005930




# import os

# print(__file__)
# print(os.path.realpath(__file__))
# print(os.path.abspath(__file__))

# load json and create model
# json_file = open('model\model2\params.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close() 
# loaded_model = model_from_json(loaded_model_json)

# # load weights into new model
# loaded_model.load_weights("model\pg_lstm_policy_20201106130656.h5")
# print("Loaded model from disk")



# # load weights into new model 

reconstructed_model = load_model("model/pg_lstm_policy_20201106130656.h5")
print("Loaded model from disk")
# # Let's check:
# # np.testing.assert_allclose(
#     model.predict(test_input), reconstructed_model.predict(test_input)
# )

# evaluate loaded model on test data 
# Define X_test & Y_test data first
# loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# score = loaded_model.evaluate(X_test, Y_test, verbose=0)
# print ("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))