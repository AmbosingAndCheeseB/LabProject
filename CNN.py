import numpy as np
import PIL.Image as img
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os


file = pd.read_csv("./label(temp).csv")
file2 = pd.read_csv("./test_label(temp).csv")

label = file.values
test_label = file2.values

label = np.array(label)
print(label.shape)
print(label, label.shape)
label = label[:, np.newaxis]

test_label = np.array(test_label)
print(test_label.shape)
print(test_label, test_label.shape)
test_label = test_label[:, np.newaxis]

def img_data(path):
  image = img.open('./'+path+'/1.png')

  x = np.array(image)
  temp1 = np.array(image)

  print("초기 이미지 행렬:", x.shape)
  x = x[np.newaxis]
  x = np.divide(x, 255)
  num = len(os.listdir('./'+path+'/')) + 1

  for i in range(2, num):
    try:
      im = img.open('./'+path+'/' + str(i) + '.png')
      temp = np.array(im)
      temp = temp.astype('float32')
      temp = np.divide(temp, 255)

      if temp1.shape == temp.shape:
        temp = temp[np.newaxis]
        x = np.concatenate((x, temp), axis=0)


    except:
      print("파일 없음" + str(i))

  return x


data = img_data("resized(temp)")
test_data = img_data("test_image(temp)")

test_data_ = tf.reshape(test_data, [-1, 48*64*4])

print("행렬 합친 후:", data.shape)

data_ = tf.reshape(data, [-1, 48*64*4])
print(data_)
temp_data = tf.reshape(data_, [-1, 64, 48, 4])
print(temp_data.shape)



batch_size = 1

dataset = tf.data.Dataset.from_tensor_slices(({"image" : data_}, label))
dataset = dataset.batch(batch_size).repeat()

iterator = dataset.make_one_shot_iterator()
next_data = iterator.get_next()

dataset2 = tf.data.Dataset.from_tensor_slices(({"image" : test_data_}, test_label))
dataset2 = dataset2.batch(batch_size).repeat()

iterator2 = dataset2.make_one_shot_iterator()
next_data2 = iterator.get_next()


def build_CNN_classifier(x):
  x_image = tf.reshape(x, [-1, 64, 48, 4])

  W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 4, 512], stddev=5e-2))
  b_conv1 = tf.Variable(tf.constant(0.1, shape=[512]))
  h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

  h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 512, 256], stddev=5e-2))
  b_conv2 = tf.Variable(tf.constant(0.1, shape=[256]))
  h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

  h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  W_conv3 = tf.Variable(tf.truncated_normal(shape=[5, 5, 256, 128], stddev=5e-2))
  b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))
  h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)

  h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  W_conv4 = tf.Variable(tf.truncated_normal(shape=[5, 5, 128, 64], stddev=5e-2))
  b_conv4 = tf.Variable(tf.constant(0.1, shape=[64]))
  h_conv4 = tf.nn.relu(tf.nn.conv2d(h_pool3, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4)

  h_pool4 = tf.nn.max_pool(h_conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  W_fc1 = tf.Variable(tf.truncated_normal(shape=[3*4*64, 1024], stddev=5e-2))
  b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

  h_pool2_flat = tf.reshape(h_pool4, [-1, 3*4*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  W_fc2 = tf.Variable(tf.truncated_normal(shape=[1024, 512], stddev=5e-2))
  b_fc2 = tf.Variable(tf.constant(0.1, shape=[512]))

  h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

  W_fc3 = tf.Variable(tf.truncated_normal(shape=[512, 256], stddev=5e-2))
  b_fc3 = tf.Variable(tf.constant(0.1, shape=[256]))

  h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

  W_output = tf.Variable(tf.truncated_normal(shape=[256, 2], stddev=5e-2))
  b_output = tf.Variable(tf.constant(0.1, shape=[2]))

  logits = tf.matmul(h_fc3, W_output) + b_output


  return logits


x = tf.placeholder(tf.float32, shape=[None, 48*64*4])
y = tf.placeholder(tf.float32, shape=[None, 2])


y_pred = build_CNN_classifier(x)

y_soft = tf.nn.softmax(y_pred)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels = y))
train_step = tf.train.AdamOptimizer(1e-6).minimize(loss)

tf.summary.scalar('loss', loss)



saver_dir = "pre_model"
saver = tf.train.Saver()
ckpt_path = os.path.join(saver_dir, "model")
ckpt = tf.train.get_checkpoint_state(saver_dir)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  merged = tf.summary.merge_all()
  tensorboard_writer = tf.summary.FileWriter('./tensorboard_log', sess.graph)
  num1 = 0

  ckpt_question = input("파라미터를 불러오시겠습니까?(Y/N) ")

  if ckpt_question in "y" or ckpt_question in "Y":
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      print("저장된 가중치 파라미터가 없습니다.")

  for i in range(10001):
    batch_X, batch_Y = sess.run(next_data)
    #print(batch_X, batch_Y)
    batch_Y = batch_Y[0,:]
    #print(batch_Y)
    if i % 10 == 0:
      loss_ = sess.run(loss, feed_dict={x: batch_X['image'], y: batch_Y})
      print("반복(Epoch): %d, loss : %f" % (i, loss_))

    _, y_pred1 = sess.run([train_step, y_pred], feed_dict={x: batch_X['image'], y: batch_Y})
    summary = sess.run(merged, feed_dict = {x :batch_X['image'], y: batch_Y})
    tensorboard_writer.add_summary(summary, i)

    if i % 1000 == 0:
      saver.save(sess, ckpt_path, global_step=i)

  count = 0
  num = 0

  for j in range(10):
    batch_X, batch_Y = sess.run(next_data)
    #print(batch_X, batch_Y)
    batch_Y = batch_Y[0, :]
    # print(batch_Y)
    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))
    print("정확도(Accuracy): %f" % (accuracy.eval(feed_dict={x:batch_X['image'], y:batch_Y})))
    _, y_soft = sess.run([train_step, y_soft], feed_dict={x: batch_X['image'], y: batch_Y})
    print(y_soft)
