import hls4ml
import tensorflow as tf
import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Activation, MaxPool1D, AvgPool1D, Flatten, Dense
from tensorflow.nn import selu
from qkeras import *
import yaml
import time

def SHO_fitting_function(params):
    amp = tf.expand_dims(params[:, 0], 1)
    w_0 = tf.expand_dims(params[:, 1], 1)
    qf  = tf.expand_dims(params[:, 2], 1)
    phi = tf.expand_dims(params[:, 3], 1)
    
    amp = tf.cast(amp, tf.complex128)
    w_0 = tf.cast(w_0, tf.complex128)
    qf = tf.cast(qf, tf.complex128)
    phi = tf.cast(phi, tf.complex128)
    frequency = tf.cast(wvec_freq, tf.complex128)
    
    numer = amp * tf.math.exp((1.j) * phi) * tf.math.square(w_0)
    den_1 = tf.math.square(frequency)
    den_2 = (1.j) * frequency * w_0 / qf
    den_3 = tf.math.square(w_0)
    
    den = den_1 - den_2 - den_3
    func = numer / den
    
    real = tf.math.real(func)
    real_scaled = tf.divide(tf.subtract(tf.cast(real, tf.float32), \
                            tf.convert_to_tensor(scaler_real.mean, dtype=tf.float32)),\
                             tf.convert_to_tensor(scaler_real.std, dtype=tf.float32))
    
    imag = tf.math.imag(func)
    imag_scaled = tf.divide(tf.subtract(tf.cast(imag, tf.complex128), \
                        tf.convert_to_tensor(scaler_imag.mean, dtype=tf.complex128)),\
                          tf.convert_to_tensor(scaler_imag.std, dtype=tf.complex128))
    
    func = tf.stack((real_scaled, tf.cast(imag_scaled, tf.float32)), axis=2)
    return func

def custom_loss(y_true, y_pred):
    y_true = SHO_fitting_function(y_true)
    unscaled_params = y_pred * tf.convert_to_tensor(np.sqrt(params_scaler.var_[0:4]), dtype=tf.float32) + tf.convert_to_tensor(params_scaler.mean_[0:4], dtype=tf.float32)
    y_pred = SHO_fitting_function(unscaled_params)
    mse = tf.math.square(y_true-y_pred)
    return tf.reduce_mean(mse, axis=-1)

def build_keras_model():
    model = tf.keras.models.load_model('Quantized-SHO-Fitting/Pre-trained-Deep-Learning-Models-For-Rapid-Analysis-Of-Piezoelectric-Hysteresis-Loops-SHO-Fitting/saved_models/autoqkeras_cnn_v1', compile=False)
    sgd = tf.keras.optimizers.SGD()
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(loss=custom_loss, optimizer=sgd)
    return model
    
def convert_to_hls():
    model = build_keras_model()
    hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    
    hls_config['Model']['Strategy'] = 'Resource'
    hls_config['Model']['ReuseFactor'] = 12 
    for layer in hls_config['LayerName'].keys():
        hls_config['LayerName'][layer]['ReuseFactor'] = 12
        hls_config['LayerName'][layer]['Strategy'] = 'Resource'
       

    #     cfg = hls4ml.converters.create_config(backend='Vivado')
    #     cfg['IOType']     = 'io_stream' # Must set this if using CNNs!
    #     cfg['HLSConfig']  = hls_config
    #     cfg['KerasModel'] = model
    #     cfg['OutputDir']  = 'quantized_cnn_v1/'
    #     cfg['XilinxPart'] = 'xcku060-1ffva1517c'
    
    #     hls_model = hls4ml.converters.keras_to_hls(cfg)

    folder_name = 'quantized_cnn_v1'
    # Convert to an hls model
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=hls_config, output_dir=folder_name)
    
    hls_model.compile()
    path = f'{folder_name}/hls4ml_config.yml'
    time.sleep(5)
    list_attrs = {}
    with open(path) as f:
        list_attrs = yaml.load(f, Loader=yaml.BaseLoader)
        list_attrs['IOType'] = 'io_stream'
        list_attrs['XilinxPart'] = 'xcku060-1ffva1517c'
     
    with open(path, 'w') as f:
        yaml.dump(list_attrs, f)
    hls_model.build(csim=False, synth=True, vsynth=True, export=True)
    
def main():
    convert_to_hls()
  
if __name__ == '__main__':
    main()
 