

from .delay import Delay
from .time_stretch import TimeStretchAudiomentation as TimeStretch
from .time_stretch import DiscreteTimeStretchAudioMentation as DiscreteTimeStretch
from .time_stretch import BetaTimeStretchAudiomentation as BetaTimeStretch
from .time_stretch import LogUniformTimeStretchAudiomentation as LogUniformTimeStretch
from .reverse import Reverse

from .pedalboard_audiomentation import PedalBoardAudiomentation

from .chorus import ChorusAudiomentation
from .compression import CompressorAudiomentation
from .distortion import DistortionAudiomentation
from .reverb import ReverbAudiomentation
from .bitcrush import BitcrushAudiomentation
from .background import AddBackgroundNoiseAudiomentation
from .custom_pitch_shift import PitchShiftAudiomentation
