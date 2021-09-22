import tensorflow as tf
import tensorflow_datasets as tfds


def download_imagenet():
    ds = tfds.load('imagenet2012',
        data_dir='~/data/tf-imagenet-dirs/data',
        split='validation',
        shuffle_files=False,
        download=False,
        as_supervised=True)
    return ds


def resize_with_crop(image, preprocessor, need_transpose):
    from tensorflow.python.keras.applications import imagenet_utils
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (256, 256))
    image = tf.image.central_crop(image, 0.875) # 224x224
    if need_transpose:
        image = image[..., ::-1]
    if type(preprocessor) == str:
        image = imagenet_utils.preprocess_input(image, mode=preprocessor)
    elif preprocessor is not None:
        image = preprocessor(image)
    return image


def quantize_model_with_imagenet(keras_model, preprocessor, need_transpose, max_i=500):
    def representative_dataset():
        ds = download_imagenet()
        for i, (image, label) in enumerate(ds):
          if i==max_i:
              break
          image = resize_with_crop(image, preprocessor, need_transpose)
          image = tf.reshape(image, (1, *image.shape))
          yield [image]

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    tflite_model = converter.convert()
    return tflite_model


def main():
    keras_model = tf.keras.applications.ResNet50(weights='imagenet')
    keras_model.summary()
    preprocessor = 'caffe'
    need_transpose = False
    tflite_model_buf = quantize_model_with_imagenet(keras_model, preprocessor, need_transpose)


if __name__=='__main__':
    main()
