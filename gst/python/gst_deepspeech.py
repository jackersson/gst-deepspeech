"""
Usage
    export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PWD/venv/lib/gstreamer-1.0/:$PWD/gst/
    export GST_DEBUG=python:4

    gst-launch-1.0 filesrc location=video.mp4 ! decodebin ! videoconvert ! \
        deepspeech ! videoconvert ! autovideosink
"""

import os
import yaml
import logging
import audioop
import trafaret as t
import typing as typ
import yaml
import numpy as np
from deepspeech import Model

from gstreamer import Gst, GObject, GstBase, GstVideo, GLib


import gstreamer.utils as utils
from gstreamer.gst_objects_info_meta import gst_meta_write

import gi
gi.require_version('GstAudio', '1.0')
from gi.repository import GstAudio  # noqa:F401,F402


def _get_log_level() -> int:
    return int(os.getenv("GST_PYTHON_LOG_LEVEL", logging.DEBUG / 10)) * 10


log = logging.getLogger('gst_python')
log.setLevel(_get_log_level())


def load_config(filename: str) -> dict:
    if not os.path.isfile(filename):
        raise ValueError(f"Invalid filename {filename}")

    with open(filename, 'r') as stream:
        try:
            data = yaml.load(stream, Loader=yaml.Loader)
            return data
        except yaml.YAMLError as exc:
            raise OSError(f'Parsing error. Filename: {filename}')


# Paper: https://arxiv.org/pdf/1412.5567.pdf
# WER (Word Error Rate): https://en.wikipedia.org/wiki/Word_error_rate
class DeepSpeechModel:

    # default configuration
    _config_schema = t.Dict({
        # path or name of the frozen model file (*.pb, *.pbmm)
        t.Key('model', default="data/models/deepspeech/output_graph.pbmm"): t.String(min_length=4),
        # Path to the language model binary file
        t.Key('language_model', default="data/models/deepspeech/lm.binary"): t.String(min_length=4),
        # trie file (build from the same vocabulary as the language model binary)
        # https://www.youtube.com/watch?v=-urNrIAQnNo
        t.Key('trie', default="data/models/deepspeech/trie"): t.String(min_length=4),

        # The alpha hyperparameter of the CTC decoder. Language Model weight.
        t.Key('lm_alpha', default=0.75): t.Float(),

        # The beta hyperparameter of the CTC decoder. Word insertion weight.
        t.Key('lm_beta', default=1.85): t.Float(),

        # A larger beam width value generates better results at the cost of decoding time.
        t.Key('beam_width', default=500): t.Int(gt=0),

    }, allow_extra='*')

    def __init__(self, **kwargs):
        # validate config
        try:
            self.config = self._config_schema.check(kwargs or {})
        except t.DataError as err:
            raise ValueError(
                'Wrong model configuration for {}: {}'.format(self, err))

        self.model = None

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return '<{}>'.format(self)

    def __enter__(self):
        self.startup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    @property
    def log(self) -> logging.Logger:
        return log

    def startup(self):

        self.log.info("Starting %s ...", self)

        if not os.path.isfile(self.config['model']):
            raise FileNotFoundError(f"Invalid filename {self.config['model']}")

        self.model = Model(self.config['model'], self.config['beam_width'])

        if os.path.isfile(self.config['language_model']) and \
                os.path.isfile(self.config['trie']):

            self.model.enableDecoderWithLM(self.config['language_model'],
                                           self.config['trie'],
                                           self.config['lm_alpha'],
                                           self.config['lm_beta'])

        self.log.info("Warming up %s ...", self)
        self.model.stt(np.zeros(160, dtype=np.int16))

    def shutdown(self):
        """ Releases model when object deleted """
        self.log.info("Shutdown %s ...", self)
        self.model = None
        self.log.info("%s Destroyed successfully", self)

    def process(self, audio: np.ndarray) -> str:
        return self.model.stt(audio)


def from_config_file(filename: str) -> DeepSpeechModel:
    """
    :param filename: filename to model config
    """
    return DeepSpeechModel(**load_config(filename))


class GstDeepSpeechPluginPy(GstBase.BaseTransform):

    GST_PLUGIN_NAME = 'deepspeech'
    DEFAULT_SILENCE_THRESHOLD = -1  # 0.1
    DEFAULT_SILENCE_DURATION = 5
    DEFAULT_MAX_NUM_FRAMES_SEQ = 150
    DEFAULT_MIN_NUM_FRAMES_SEQ = 50

    # Metadata Explanation:
    # http://lifestyletransfer.com/how-to-create-simple-blurfilter-with-gstreamer-in-python-using-opencv/
    __gstmetadata__ = ("Name",
                       "Transform",
                       "Description",
                       "Author")

    _srctemplate = Gst.PadTemplate.new('src', Gst.PadDirection.SRC,
                                       Gst.PadPresence.ALWAYS,
                                       Gst.Caps.from_string("audio/x-raw,format=S16LE,rate=16000,channels=1"))

    _sinktemplate = Gst.PadTemplate.new('sink', Gst.PadDirection.SINK,
                                        Gst.PadPresence.ALWAYS,
                                        Gst.Caps.from_string("audio/x-raw,format=S16LE,rate=16000,channels=1"))

    __gsttemplates__ = (_srctemplate, _sinktemplate)

    # Explanation: https://python-gtk-3-tutorial.readthedocs.io/en/latest/objects.html#GObject.GObject.__gproperties__
    # Example: https://python-gtk-3-tutorial.readthedocs.io/en/latest/objects.html#properties
    __gproperties__ = {
        "model": (GObject.TYPE_PYOBJECT,
                  "model",
                  "Contains model DeepSpeechModel",
                  GObject.ParamFlags.READWRITE),

        "config": (str,
                   "Path to config file",
                   "Contains path to config *.yaml supported by DeepSpeechModel",
                   None,  # default
                   GObject.ParamFlags.READWRITE
                   ),

        "silence_threshold": (GObject.TYPE_FLOAT,
                              "Silence threshold",
                              "Detect silence if volume is lower than silence threshold. Disabled: -1",
                              -1,  # min
                              GLib.MAXINT,  # max
                              DEFAULT_SILENCE_THRESHOLD,  # default
                              GObject.ParamFlags.READWRITE
                              ),

        "silence_duration": (GObject.TYPE_INT,
                             "Silence Duration",
                             "Detect silence if number ob buffers more than silence duration.",
                             -1,  # min
                             GLib.MAXINT,  # max
                             DEFAULT_SILENCE_THRESHOLD,  # default
                             GObject.ParamFlags.READWRITE
                             ),


        "max_num_frames_seq": (GObject.TYPE_UINT,
                               "Maximum number of frames sequence",
                               "Do speech recognition when num_frames >= max_num_frames_seq. Default: 150",
                               0,  # min
                               GLib.MAXUINT,  # max
                               DEFAULT_MAX_NUM_FRAMES_SEQ,  # default
                               GObject.ParamFlags.READWRITE
                               ),

        "min_num_frames_seq": (GObject.TYPE_UINT,
                               "Minimum number of frames sequence",
                               "Wait for until num_frames >= min_num_frames_seq to process. Default: 50",
                               0,  # min
                               GLib.MAXUINT,  # max
                               DEFAULT_MIN_NUM_FRAMES_SEQ,  # default
                               GObject.ParamFlags.READWRITE
                               ),
    }

    def __init__(self):
        super().__init__()

        self._frames = []

        self._silence_threshold = self.DEFAULT_SILENCE_THRESHOLD
        self._silence_duration = self.DEFAULT_SILENCE_DURATION
        self._max_num_frames_seq = self.DEFAULT_MAX_NUM_FRAMES_SEQ
        self._min_num_frames_seq = self.DEFAULT_MIN_NUM_FRAMES_SEQ

        self._model = None
        self._config = None

        # self._amplitude_mean = 0
        self._silent_buffers_num = 0

        # self.model_path = "data/models/deepspeech/output_graph.pbmm"

        # if gpu
        # model_path = "data/models/deepspeech/output_graph.pbmm"

        # if cpu
        # model_path = "data/models/deepspeech/output_graph.pbmm"

        # self.model = Model(model_path, 500)
        # self.model.stt(np.zeros(160, dtype=np.int16))
        # model.enableDecoderWithLM(ARGS.lm, ARGS.trie, ARGS.lm_alpha, ARGS.lm_beta)

    def do_transform_ip(self, buffer: Gst.Buffer) -> Gst.FlowReturn:

        if not self._model:
            Gst.warning(f"No model speficied for {self}. Plugin working in passthrough mode")
            return Gst.FlowReturn.OK

        # print(buffer.get_size(), buffer.duration / 10**6)

        # map Gst.Buffer to READ content
        is_ok, map_info = buffer.map(Gst.MapFlags.READ)
        if is_ok:
            # parsing audio info
            # https://lazka.github.io/pgi-docs/GstAudio-1.0/classes/AudioInfo.html
            audio_info = GstAudio.AudioInfo()
            audio_info.from_caps(self.sinkpad.get_current_caps())

            # bpf = bytes per frame (for S16LE bpf = 2 bytes)
            # np.int16 -> due to audio format S16LE
            # https://lazka.github.io/pgi-docs/GstAudio-1.0/enums.html#GstAudio.AudioFormat.S16LE
            frame = np.ndarray(map_info.size // audio_info.bpf,
                               buffer=map_info.data,
                               dtype=np.int16)

            self._frames.append(frame)
            buffer.unmap(map_info)

            cut_on_silence = self._silence_threshold > 0
            num_frames = len(self._frames)
            if num_frames >= self._max_num_frames_seq:
                self._do_speech_recognition()
            else:
                if cut_on_silence and num_frames >= self._min_num_frames_seq:
                    square = np.sqr(frame)
                    peaksquare = np.max(square)
                    squaresum = np.sum(square)

                    normalizer = 1 << 30
                    ncs = float(squaresum) / normalizer

                    print(ncs)

                    if ncs > self._silence_threshold:
                        self._silent_buffers_num += 1

                    if self._silent_buffers_num >= self._silence_duration:
                        self._do_speech_recognition()

                    # self._amplitude_mean += np.mean(frame)
                    # self._amplitude_mean /= 2
                    # rms = np.sqrt(self._amplitude_mean ** 2)

                    # if num_frames >= self._min_num_frames_seq and rms < self._rms_threshold:
                    #     self._do_speech_recognition()

        return Gst.FlowReturn.OK

    def _do_speech_recognition(self):
        print(self._model.process(np.hstack(self._frames)))

        self._frames = []
        self._silent_buffers_num = 0

    def do_get_property(self, prop: GObject.GParamSpec):
        if prop.name == 'model':
            return self._model
        if prop.name == 'config':
            return self._config
        else:
            raise AttributeError('Unknown property %s' % prop.name)

    def do_set_property(self, prop: GObject.GParamSpec, value):
        if prop.name == 'model':
            self._do_set_model(value)
        elif prop.name == "config":
            self._do_set_model(from_config_file(value))
            self._config = value
            Gst.info(f"Model's config updated from {self._config}")
        else:
            raise AttributeError('Unknown property %s' % prop.name)

    def _do_set_model(self, model: DeepSpeechModel):

        # stop previous instance
        if self._model:
            self._model.shutdown()
            self._model = None

        self._model = model

        # start new instance
        if self._model:
            self._model.startup()


# Required for registering plugin dynamically
# Explained: http://lifestyletransfer.com/how-to-write-gstreamer-plugin-with-python/
GObject.type_register(GstDeepSpeechPluginPy)
__gstelementfactory__ = (GstDeepSpeechPluginPy.GST_PLUGIN_NAME,
                         Gst.Rank.NONE, GstDeepSpeechPluginPy)


# if self.counter >= self.num_frames or self.counter == 0:

#     if isinstance(self.frame, np.ndarray):
#         # n = self.frame.shape[0]
#         # k = audio.shape[0] * 2
#         # from_index = np.clip(n - k, 0, n - 1)
#         # to_index = np.clip(from_index + k, from_index, n - 1)

#         # # print((to_index - from_index) // audio.shape[0], from_index, to_index)

#         # chunk = self.frame[from_index:to_index]
#         rms = np.sqrt(np.mean(audio) ** 2)
#         if self.rms_threshold < 0 or rms < self.rms_threshold:
#             processed_data = self.model.stt(self.frame)
#             print(processed_data)

#             # print(self.frame.shape if isinstance(self.frame, np.ndarray) else None)
#             self.frame = self.frame[-2:]
#             self.counter = 0
#     else:
#         self.frame = audio
#         self.counter = 0
# else:
#     self.frame = np.concatenate((self.frame, audio))
#     # processed_data = self.model.stt(audio)
#     # print(np.mean(audio), processed_data, "h")
#     # print(processed_data)
# self.counter += 1

# print(len(self.frame))


 # in_info.from_caps(self.sinkpad.get_current_caps())
# print(in_info.finfo.depth, in_info.bpf)

# print(self.sinkpad.get_current_caps())

# if self.model is None:
#     Gst.warning(f"No model speficied for {self}. Plugin working in passthrough mode")
#     return Gst.FlowReturn.OK

# try:
#     # Convert Gst.Buffer to np.ndarray
#     image = utils.gst_buffer_with_caps_to_ndarray(buffer, self.sinkpad.get_current_caps())

#     # model inference
#     objects = self.model.process_single(image)

#     Gst.debug(f"Frame id ({buffer.pts // buffer.duration}). Detected {str(objects)}")

#     # write objects to as Gst.Buffer's metadata
#     # Explained: http://lifestyletransfer.com/how-to-add-metadata-to-gstreamer-buffer-in-python/
#     gst_meta_write(buffer, objects)
# except Exception as err:
#     logging.error("Error %s: %s", self, err)


# print(len(audio), map_info.size, in_info.channels, in_info.rate)
# rms = np.sqrt(np.mean(audio) ** 2)
# print(rms)

# amplitude_squared = 10 * np.log(np.mean(audio) ** 2)
# return Gst.FlowReturn.OK
