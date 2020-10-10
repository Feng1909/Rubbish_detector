import os
import json

# 设置要生成文件的路径
data_root_path = './dataset'
# 所有类别的信息
class_detail = []
# 获取所有类别保存的文件夹名称，这里是[daisy，dandelion，roses，sunflowers]
class_dirs = os.listdir(data_root_path)
try:
    class_dirs.remove('test.list')
    class_dirs.remove('trainer.list')
    class_dirs.remove('readme.json')

    class_dirs.remove('.trainer.list.swp')
    class_dirs.remove('.ipynb_checkpoints')
except BaseException:
    print("success")
else:
    print("success")

# 类别标签
class_label = 0
# 获取总类别的名称
father_paths = data_root_path.split('/')    #['', 'home', 'aistudio', 'data', 'data2394', 'images', 'face']
while True:
    if father_paths[father_paths.__len__() - 1] == '':
        del father_paths[father_paths.__len__() - 1]
    else:
        break
father_path = father_paths[father_paths.__len__() - 1]
print('path:',father_path)
# 把生产的数据列表都放在自己的总类别文件夹中
data_list_path = './dataset/'
# 如果不存在这个文件夹,就创建
isexist = os.path.exists(data_list_path)
if not isexist:
    os.makedirs(data_list_path)
print ('开始生成数据！')
# 清空原来的数据
with open(data_list_path + "test.list", 'w') as f:
    pass
with open(data_list_path + "trainer.list", 'w') as f:
    pass
# 总的图像数量
all_class_images = 0
# 读取每个类别
print(class_dirs)
for class_dir in class_dirs:
    # 每个类别的信息
    class_detail_list = {}
    test_sum = 0
    trainer_sum = 0
    # 统计每个类别有多少张图片
    class_sum = 0
    # 获取类别路径
    path = data_root_path + "/" + class_dir
    # 获取所有图片
    img_paths = os.listdir(path)

    for img_path in img_paths:                                  # 遍历文件夹下的每个图片
        name_path = path + '/' + img_path                       # 每张图片的路径
        if class_sum % 10 == 0:                                 # 每10张图片取一个做测试数据
            test_sum += 1                                       #test_sum测试数据的数目
            with open(data_list_path + "test.list", 'a') as f:
                f.write(name_path + "\t%d" % class_label + "\n") #class_label 标签：0,1,2
        else:
            trainer_sum += 1                                    #trainer_sum测试数据的数目
            with open(data_list_path + "trainer.list", 'a') as f:
                f.write(name_path + "\t%d" % class_label + "\n")#class_label 标签：0,1,2
        class_sum += 1                                          #每类图片的数目
        all_class_images += 1                                   #所有类图片的数目

    # 说明的json文件的class_detail数据
    class_detail_list['class_name'] = class_dir             #类别名称，如jiangwen
    class_detail_list['class_label'] = class_label          #类别标签，0,1,2
    class_detail_list['class_test_images'] = test_sum       #该类数据的测试集数目
    class_detail_list['class_trainer_images'] = trainer_sum #该类数据的训练集数目
    class_detail.append(class_detail_list)
    class_label += 1                                            #class_label 标签：0,1,2
# 获取类别数量
all_class_sum = class_dirs.__len__()
# 说明的json文件信息
readjson = {}
readjson['all_class_name'] = father_path                  #文件父目录
readjson['all_class_sum'] = all_class_sum                #
readjson['all_class_images'] = all_class_images
readjson['class_detail'] = class_detail
jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(',', ': '))
with open(data_list_path + "readme.json",'w') as f:
    f.write(jsons)
print ('生成数据列表完成！')


#识别的五大步骤：准备数据、配置网络，训练网络，模型评估、模型预测
#导入要用到的模块
import paddle
import paddle.fluid as fluid
import numpy
import sys
from multiprocessing import cpu_count


# 准备数据
# 自定义数据集需要先定义自己的reader，把图像数据处理一些，并输出图片的数组和标签。
# 定义训练的mapper
# train_mapper函数的作用是用来对训练集的图像进行处理修剪和数组变换，返回img数组和标签
# sample是一个python元组，里面保存着图片的地址和标签。 ('图片路径', 标签)
def train_mapper(sample):
    img, label = sample
    # 进行图片的读取，由于数据集的像素维度各不相同，需要进一步处理对图像进行变换
    img = paddle.dataset.image.load_image(img)
    # 进行了简单的图像变换，这里对图像进行crop修剪操作，输出img的维度为(3, 224, 224)
    img = paddle.dataset.image.simple_transform(im=img,  # 输入图片是HWC
                                                resize_size=224,  # 剪裁图片
                                                crop_size=224,
                                                is_color=True,  # 彩色图像
                                                is_train=True)
    # 将img数组进行进行归一化处理，得到0到1之间的数值
    img = img.flatten().astype('float32') / 255.0
    return img, label


# 对自定义数据集创建训练集train的reader
def train_r(train_list, buffered_size=1024):
    def reader():
        with open(train_list, 'r') as f:
            # 将train.list里面的标签和图片的地址方法一个list列表里面，中间用\t隔开'

            lines = [line.strip() for line in f]
            for line in lines:
                # 图像的路径和标签是以\t来分割的,所以我们在生成这个列表的时候,使用\t就可以了
                img_path, lab = line.strip().split('\t')
                yield img_path, int(lab)  # 每执行到一个 yield 语句就会中断，并返回一个迭代值，下次执行时从 yield 的下一个语句继续执行，每次中断都会通过 yield 返回当前的迭代值

    # 创建自定义数据训练集的train_reader
    return paddle.reader.xmap_readers(train_mapper, reader, cpu_count(), buffered_size)


# sample是一个python元组，里面保存着图片的地址和标签。 ('../images/face/zhangziyi/20181206145348.png', 2)
def test_mapper(sample):
    img, label = sample
    img = paddle.dataset.image.load_image(img)
    img = paddle.dataset.image.simple_transform(im=img, resize_size=224, crop_size=224, is_color=True, is_train=False)
    img = img.flatten().astype('float32') / 255.0
    return img, label


# 对自定义数据集创建验证集test的reader
def test_r(test_list, buffered_size=1024):
    def reader():
        with open(test_list, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                # 图像的路径和标签是以\t来分割的,所以我们在生成这个列表的时候,使用\t就可以了
                img_path, lab = line.strip().split('\t')
                yield img_path, int(lab)

    return paddle.reader.xmap_readers(test_mapper, reader, cpu_count(), buffered_size)


#读入数据（size可调）
BATCH_SIZE = 16
# 把图片数据生成reader注意换成自己的路径
trainer_reader = train_r(train_list="./dataset/trainer.list")
print('begin read train')
train_reader = paddle.batch(
    paddle.reader.shuffle(
        reader=trainer_reader,buf_size=300),
    batch_size=BATCH_SIZE)
#尝试打印一下，观察一下自定义的数据集
temp_reader = paddle.batch(trainer_reader,
                            batch_size=3)
temp_data=next(temp_reader())
print(temp_data)

print('begin read test')
tester_reader = test_r(test_list="dataset/test.list")
test_reader = paddle.batch(
     tester_reader, batch_size=BATCH_SIZE)


image = fluid.layers.data(name='image', shape=[3, 224, 224], dtype='float32')#[3, 224, 224]，表示为三通道，224*224的RGB图
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
print('image_shape:',image.shape)


def convolutional_neural_network(img):
    # 第一个卷积--池化层 模型参数卷积核池化层步长可调
    conv_pool_1 = fluid.nets.simple_img_conv_pool(input=img,  # 输入图像
                                                  filter_size=3,  # 滤波器的大小
                                                  num_filters=32,  # filter 的数量。它与输出的通道相同
                                                  pool_size=2,  # 池化层大小2*2
                                                  pool_stride=2,  # 池化层步长
                                                  act='relu')  # 激活类型

    # Dropout主要作用是减少过拟合，随机让某些权重不更新
    # Dropout是一种正则化技术，通过在训练过程中阻止神经元节点间的联合适应性来减少过拟合。
    # 根据给定的丢弃概率dropout随机将一些神经元输出设置为0，其他的仍保持不变。
    # drop = fluid.layers.dropout(x=conv_pool_1, dropout_prob=0.5)

    # 第二个卷积--池化层
    conv_pool_2 = fluid.nets.simple_img_conv_pool(input=conv_pool_1,
                                                  filter_size=3,
                                                  num_filters=64,
                                                  pool_size=2,
                                                  pool_stride=2,
                                                  act='relu')
    # 减少过拟合，随机让某些权重不更新
    # drop = fluid.layers.dropout(x=conv_pool_2, dropout_prob=0.5)

    # 第三个卷积--池化层
    conv_pool_3 = fluid.nets.simple_img_conv_pool(input=conv_pool_2,
                                                  filter_size=3,
                                                  num_filters=64,
                                                  pool_size=2,
                                                  pool_stride=2,
                                                  act='relu')
    # 减少过拟合，随机让某些权重不更新
    # drop = fluid.layers.dropout(x=conv_pool_3, dropout_prob=0.5)

    # 全连接层
    predict = fluid.layers.fc(input=conv_pool_3, size=10, act='softmax')
    # 减少过拟合，随机让某些权重不更新
    # drop =  fluid.layers.dropout(x=fc, dropout_prob=0.5)
    return predict

predict = convolutional_neural_network(image) #LeNet5卷积神经网络
# 获取损失函数和准确率
cost = fluid.layers.cross_entropy(input=predict, label=label)
# 计算cost中所有元素的平均值
avg_cost = fluid.layers.mean(cost)
#计算准确率
accuracy = fluid.layers.accuracy(input=predict, label=label)
# 定义优化方法
optimizer = fluid.optimizer.Adam(learning_rate=0.0001)# Adam是一阶基于梯度下降的算法，基于自适应低阶矩估计该函数实现了自适应矩估计优化器
optimizer.minimize(avg_cost)# 取局部最优化的平均损失
print(type(accuracy))

#训练分为三步：第一步配置好训练的环境，第二步用训练集进行训练，并用验证集对训练进行评估，不断优化，第三步保存好训练的模型
# 使用CPU进行训练
place = fluid.CUDAPlace(0)
# 创建一个executor
exe = fluid.Executor(place)
# 对program进行参数初始化1.网络模型2.损失函数3.优化函数
exe.run(fluid.default_startup_program())
# 定义输入数据的维度,DataFeeder 负责将reader(读取器)返回的数据转成一种特殊的数据结构，使它们可以输入到 Executor
feeder = fluid.DataFeeder(feed_list=[image, label], place=place)#定义输入数据的维度，第一个是图片数据，第二个是图片对应的标签。

# 这次训练2个Pass。每一个Pass训练结束之后，再使用验证集进行验证，并求出相应的损失值Cost和准确率acc。
# 训练的轮数
EPOCH_NUM = 30
print('开始训练...')
for pass_id in range(EPOCH_NUM):
    train_cost = 0
    for batch_id, data in enumerate(train_reader()):  # 遍历train_reader的迭代器，并为数据加上索引batch_id
        train_cost, train_acc = exe.run(
            program=fluid.default_main_program(),  # 运行主程序
            feed=feeder.feed(data),  # 喂入一个batch的数据
            fetch_list=[avg_cost, accuracy])  # fetch均方误差和准确率
        if batch_id % 10 == 0:  # 每10次batch打印一次训练、进行一次测试
            print("\nPass %d, Step %d, Cost %f, Acc %f" % (pass_id, batch_id, train_cost[0], train_acc[0]))
    # 开始测试
    test_accs = []  # 测试的损失值
    test_costs = []  # 测试的准确率

    # 每训练一轮 进行一次测试
    for batch_id, data in enumerate(test_reader()):  # 遍历test_reader
        test_cost, test_acc = exe.run(program=fluid.default_main_program(),  # #运行测试主程序
                                      feed=feeder.feed(data),  # 喂入一个batch的数据
                                      fetch_list=[avg_cost, accuracy])  # fetch均方误差、准确率
        test_accs.append(test_acc[0])  # 记录每个batch的误差
        test_costs.append(test_cost[0])  # 记录每个batch的准确率

    test_cost = (sum(test_costs) / len(test_costs))  # 每轮的平均误差
    test_acc = (sum(test_accs) / len(test_accs))  # 每轮的平均准确率
    print('Test:%d, Cost:%0.5f, ACC:%0.5f' % (pass_id, test_cost, test_acc))

    # 两种方法，用两个不同的路径分别保存训练的模型
    model_save_dir = "model_soft"
    # model_save_dir = "/home/aistudio/data/data2815/model_mlp"
    # model_save_dir = "/home/aistudio/data/data2815/model_cnn"
    # 如果保存路径不存在就创建
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # 保存训练的模型，executor 把所有相关参数保存到 dirname 中
    fluid.io.save_inference_model(dirname=model_save_dir, feeded_var_names=["image"], target_vars=[predict],
                                  executor=exe)

print('训练模型保存完成！')