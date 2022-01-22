from library import *

def power(waveform, SR):
    waveform = np.sum(np.square(waveform)) / (2*(SR)+1)
    return waveform

def SNR(signal, noise):
    return 20*np.log(signal/noise)




# def adding_noise(orginal_signal, noise_signal):
#     print(noise_signal.shape[0]*0.2)
#     return orginal_signal + 

def read_signal(signal):
    data, _= librosa.load(signal)
    return data

def get_label(path, isInclude=True):
    return path.split("\\")[-2] if isInclude else path.split("\\")[-1]

def load_data(path):
    return librosa.load(path)

def load_noise(paths, sample):
    i = np.random.choice(len(paths), 1)[0]
    noise,sr = librosa.load(paths)
    noise_samples = noise.shape[0]
    noise_samples = noise_samples//2
    de = noise_samples-sample//2
    return noise[de:de+sample]

def STFT(signal):
    spectrogram = tf.signal.stft(
        signal, frame_length=255, frame_step=128)
    
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def get_data_label(path,noise_path, input_length = 25000):
    label = get_label(path)
    print(label)
    data,sr = load_data(path)
    zeros = np.zeros(input_length - data.shape[0])
    data = np.concatenate((data, zeros),axis=None)
    noise = load_noise(noise_path, data.shape[0])
    data = data + noise
    spectugram_data = STFT(data)
    return label,data,spectugram_data

def hot_encoding(a,labels,label):
    return a[labels.index(label)]

def mel_spectrum():
    pass 

def MFCC():
    pass

def model(shape, classes):
    inp = Input(shape=shape)
    filters = [128,64,32]
    dilation = [2,1,2]
    x = experimental.preprocessing.Resizing(32, 32)(inp)
    for i in range(len(filters)):
        x = Conv2D(
                filters=filters[i],
                kernel_size=3,
                dilation_rate=(dilation[i],dilation[i]),
                padding="same", 
                activation="relu")(x)

        x = BatchNormalization()(x)
        x = MaxPool2D()(x)
        
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(classes, activation="softmax")(x)
    model = Model(inputs=inp, outputs=x)
    return model

def split(data, precentage, seed):
    random.seed(seed)
    random.shuffle(data)
    precentage = int(precentage * len(data))

    train = np.array(data[:precentage])
    test = np.array(data[precentage:])

    train_label = train[:,0]
    test_label = test[:,0]
    train_data = list(train[:,2])
    test_data = list(test[:,2])

    print("label sdfasd asdf asdf", train_label)
    return train_data,test_data,train_label, test_label
