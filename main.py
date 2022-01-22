from library import *
from functions import *


config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.9
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = True
device = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(device[0], True)
run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)


# a = np.random.choice()
PATH = "D:\\Data\\speach recognation\\train\\audio\\"
sound_path = PATH + "bed\\00f0204f_nohash_0.wav"
sound_path1 = PATH + "bed\\0b56bcfe_nohash_0.wav"
noise_path = "D:\\Data\\speach recognation\\train\\_background_noise_"

paths = glob.glob(PATH + "*/*")
labels = glob.glob(PATH +"*" )
paths_noise = glob.glob(noise_path+"/*")
paths_noise.remove(r"D:\Data\speach recognation\train\_background_noise_\README.md")

number_of_path = len(paths)
number_of_noise = len(paths_noise)
a = number_of_path//number_of_noise
paths_noise = a*paths_noise
last = len(paths) - len(paths_noise)
paths_noise = paths_noise + paths_noise[:last]

random.shuffle(paths)
random.shuffle(paths_noise)

fraction_of_data = int(number_of_path * 0.1)
all_data = [*map(get_data_label, paths[:fraction_of_data], paths_noise[:fraction_of_data])]
# all_data = list(all_data)
lab,da, spe = all_data[0]

if len(spe.shape) > 2:
    assert len(spe.shape) == 3
    spectrogram = np.squeeze(spe, axis=-1)

log_spec = np.log(spectrogram.T + np.finfo(float).eps)
height = log_spec.shape[0]
width = log_spec.shape[1]
print(log_spec.shape)
X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
print(np.size(spectrogram))
Y = range(height)
print(Y)


print(spectrogram.shape)
plt.subplot(2,1,1)
plt.plot(da)
plt.subplot(2,1,2)
plt.pcolormesh(X,Y,log_spec)
plt.title(lab)
plt.show()
data,sr1 = librosa.load(sound_path)
data1,sr1 = librosa.load(sound_path1)
print(data.shape)
print(data1.shape)
# noise_data,sr2 = librosa.load(noise_path)
# print(data.shape, noise_data.shape)
# print(sr1 , sr2)
# # playsound(path)
# signal_power = power(data,sr1)
# noise_power = power(noise_data, sr2)
labels = map(get_label, labels, [False]*len(labels))
labels = list(labels)
a = np.eye(len(labels))
print(labels)
print(f"string lable {lab}, hot encoding {hot_encoding(a,labels,lab)}")

train_data,test_data,train_label,test_label = split(all_data,0.7,52)
print("this all data",train_label)

train_label = np.array([hot_encoding(a,labels,i) for i in train_label])
test_label = np.array([hot_encoding(a,labels,i) for i in test_label])

classes = len(labels)
input_shape = spe.shape
# label = np.frombuffer(label,dtype=np.dtype(str))
print(spe.shape)
mode = model(input_shape, classes)
print(mode.summary())

mode.compile(loss="CategoricalCrossentropy", optimizer="Adam", metrics="accuracy")

batch_size = 8
epochs = 100

train_data = tf.convert_to_tensor(train_data)
test_data = tf.convert_to_tensor(test_data)
print(train_data.shape)

callback =tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                      patience=5,
                      verbose=0),
# test_data = np.asarray(test_data).astype('float32')
mode.fit(
    train_data,train_label,
    validation_data=(test_data,test_label),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[callback]
)

# print(SNR(signal_power, noise_power))
# signal = adding_noise(data,noise_data)
# all_signal = [data, noise_data, signal]
# plt.figure(figsize=(10,12))
# for i in range(3):
#     plt.subplot(1, 3, i+1)
#     plt.specgram(all_signal[i],)
# plt.show()



# write('combined signal.wav', sr1, signal)
# playsound('combined signal.wav')
# librosa.display.waveplot(signal, alpha=0.5)
# plt.show()



# classes 

# class DataPreprocessing:
#     def __init__(self, data_paths,noise_paths) -> None:
#         self.data_paths = data_paths
#         self.noise_paths = noise_paths
#         self.label = []
#         self.data = []

#     def get_labels_data(slef):
        

#     def labels(self):