# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.1
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _iarius
else:
    import _iarius

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


PGA_GAIN_PGA_GAIN_24dB = _iarius.PGA_GAIN_PGA_GAIN_24dB
PGA_GAIN_PGA_GAIN_30dB = _iarius.PGA_GAIN_PGA_GAIN_30dB
LPF_PROG_LPF_PROG_15MHz = _iarius.LPF_PROG_LPF_PROG_15MHz
LPF_PROG_LPF_PROG_20MHz = _iarius.LPF_PROG_LPF_PROG_20MHz
LPF_PROG_LPF_PROG_35MHz = _iarius.LPF_PROG_LPF_PROG_35MHz
LPF_PROG_LPF_PROG_30MHz = _iarius.LPF_PROG_LPF_PROG_30MHz
LPF_PROG_LPF_PROG_50MHz = _iarius.LPF_PROG_LPF_PROG_50MHz
LPF_PROG_LPF_PROG_10MHz = _iarius.LPF_PROG_LPF_PROG_10MHz
ACTIVE_TERM_EN_ACTIVE_TERM_DIS = _iarius.ACTIVE_TERM_EN_ACTIVE_TERM_DIS
ACTIVE_TERM_EN_ACTIVE_TERM_EN = _iarius.ACTIVE_TERM_EN_ACTIVE_TERM_EN
GBL_ACTIVE_TERM_GBL_ACTIVE_TERM_50 = _iarius.GBL_ACTIVE_TERM_GBL_ACTIVE_TERM_50
GBL_ACTIVE_TERM_GBL_ACTIVE_TERM_100 = _iarius.GBL_ACTIVE_TERM_GBL_ACTIVE_TERM_100
GBL_ACTIVE_TERM_GBL_ACTIVE_TERM_200 = _iarius.GBL_ACTIVE_TERM_GBL_ACTIVE_TERM_200
GBL_ACTIVE_TERM_GBL_ACTIVE_TERM_400 = _iarius.GBL_ACTIVE_TERM_GBL_ACTIVE_TERM_400
LNA_GAIN_GBL_LNA_GAIN_GBL_18dB = _iarius.LNA_GAIN_GBL_LNA_GAIN_GBL_18dB
LNA_GAIN_GBL_LNA_GAIN_GBL_24dB = _iarius.LNA_GAIN_GBL_LNA_GAIN_GBL_24dB
LNA_GAIN_GBL_LNA_GAIN_GBL_12dB = _iarius.LNA_GAIN_GBL_LNA_GAIN_GBL_12dB
LNA_HPF_PROG_LNA_HPF_PROG_100kHz = _iarius.LNA_HPF_PROG_LNA_HPF_PROG_100kHz
LNA_HPF_PROG_LNA_HPF_PROG_50kHz = _iarius.LNA_HPF_PROG_LNA_HPF_PROG_50kHz
LNA_HPF_PROG_LNA_HPF_PROG_200kHz = _iarius.LNA_HPF_PROG_LNA_HPF_PROG_200kHz
LNA_HPF_PROG_LNA_HPF_PROG_150kHz = _iarius.LNA_HPF_PROG_LNA_HPF_PROG_150kHz
DIG_TGC_ATTENUATION_DIG_TGC_ATTENUATION_0dB = _iarius.DIG_TGC_ATTENUATION_DIG_TGC_ATTENUATION_0dB
DIG_TGC_ATTENUATION_DIG_TGC_ATTENUATION_6dB = _iarius.DIG_TGC_ATTENUATION_DIG_TGC_ATTENUATION_6dB
DIG_TGC_ATTENUATION_DIG_TGC_ATTENUATION_12dB = _iarius.DIG_TGC_ATTENUATION_DIG_TGC_ATTENUATION_12dB
DIG_TGC_ATTENUATION_DIG_TGC_ATTENUATION_18dB = _iarius.DIG_TGC_ATTENUATION_DIG_TGC_ATTENUATION_18dB
DIG_TGC_ATTENUATION_DIG_TGC_ATTENUATION_24dB = _iarius.DIG_TGC_ATTENUATION_DIG_TGC_ATTENUATION_24dB
DIG_TGC_ATTENUATION_DIG_TGC_ATTENUATION_30dB = _iarius.DIG_TGC_ATTENUATION_DIG_TGC_ATTENUATION_30dB
DIG_TGC_ATTENUATION_DIG_TGC_ATTENUATION_36dB = _iarius.DIG_TGC_ATTENUATION_DIG_TGC_ATTENUATION_36dB
DIG_TGC_ATTENUATION_DIG_TGC_ATTENUATION_42dB = _iarius.DIG_TGC_ATTENUATION_DIG_TGC_ATTENUATION_42dB
EN_DIG_TGC_EN_DIG_TGC_DIS = _iarius.EN_DIG_TGC_EN_DIG_TGC_DIS
EN_DIG_TGC_EN_DIG_TGC_EN = _iarius.EN_DIG_TGC_EN_DIG_TGC_EN
class IArius(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _iarius.delete_IArius

    def SWTrigger(self):
        return _iarius.IArius_SWTrigger(self)

    def IsPowereddown(self):
        return _iarius.IArius_IsPowereddown(self)

    def Powerup(self):
        return _iarius.IArius_Powerup(self)

    def InitializeClocks(self):
        return _iarius.IArius_InitializeClocks(self)

    def InitializeRX(self):
        return _iarius.IArius_InitializeRX(self)

    def InitializeDDR4(self):
        return _iarius.IArius_InitializeDDR4(self)

    def SyncClocks(self):
        return _iarius.IArius_SyncClocks(self)

    def ScheduleReceive(self, address, length, callback=None):
        return _iarius.IArius_ScheduleReceive(self, address, length, callback)

    def EnableReceive(self):
        return _iarius.IArius_EnableReceive(self)

    def ClearScheduledReceive(self):
        return _iarius.IArius_ClearScheduledReceive(self)

    def TransferRXBufferToHost(self, dstAddress, length, srcAddress):
        return _iarius.IArius_TransferRXBufferToHost(self, dstAddress, length, srcAddress)

    def LockDMABuffer(self, address, length):
        return _iarius.IArius_LockDMABuffer(self, address, length)

    def ReleaseDMABuffer(self, address):
        return _iarius.IArius_ReleaseDMABuffer(self, address)

    def EnableTestPatterns(self):
        return _iarius.IArius_EnableTestPatterns(self)

    def DisableTestPatterns(self):
        return _iarius.IArius_DisableTestPatterns(self)

    def SyncTestPatterns(self):
        return _iarius.IArius_SyncTestPatterns(self)

    def SetPGAGain(self, gain):
        return _iarius.IArius_SetPGAGain(self, gain)

    def SetLPFCutoff(self, cutoff):
        return _iarius.IArius_SetLPFCutoff(self, cutoff)

    def SetActiveTermination(self, endis, term):
        return _iarius.IArius_SetActiveTermination(self, endis, term)

    def SetLNAGain(self, gain):
        return _iarius.IArius_SetLNAGain(self, gain)

    def SetDTGC(self, endis, att):
        return _iarius.IArius_SetDTGC(self, endis, att)

    def InitializeTX(self):
        return _iarius.IArius_InitializeTX(self)

    def SWNextTX(self):
        return _iarius.IArius_SWNextTX(self)

    def GetID(self):
        return _iarius.IArius_GetID(self)

    def SetTxDelay(self, *args):
        return _iarius.IArius_SetTxDelay(self, *args)

    def SetTxFreqency(self, *args):
        return _iarius.IArius_SetTxFreqency(self, *args)

    def SetTxPeriods(self, *args):
        return _iarius.IArius_SetTxPeriods(self, *args)

    def SetRxAperture(self, *args):
        return _iarius.IArius_SetRxAperture(self, *args)

    def SetTxAperture(self, *args):
        return _iarius.IArius_SetTxAperture(self, *args)

    def SetRxTime(self, *args):
        return _iarius.IArius_SetRxTime(self, *args)

    def SetNumberOfFirings(self, nFirings):
        return _iarius.IArius_SetNumberOfFirings(self, nFirings)

    def EnableTransmit(self):
        return _iarius.IArius_EnableTransmit(self)

    def SetRxChannelMapping(self, srcChannel, dstChannel):
        return _iarius.IArius_SetRxChannelMapping(self, srcChannel, dstChannel)

    def SetTxChannelMapping(self, srcChannel, dstChannel):
        return _iarius.IArius_SetTxChannelMapping(self, srcChannel, dstChannel)

    def TGCEnable(self):
        return _iarius.IArius_TGCEnable(self)

    def TGCDisable(self):
        return _iarius.IArius_TGCDisable(self)

    def TGCSetSamples(self, samples, nSamples):
        return _iarius.IArius_TGCSetSamples(self, samples, nSamples)

    def TriggerStart(self):
        return _iarius.IArius_TriggerStart(self)

    def TriggerStop(self):
        return _iarius.IArius_TriggerStop(self)

    def TriggerSync(self):
        return _iarius.IArius_TriggerSync(self)

    def SetNTriggers(self, n):
        return _iarius.IArius_SetNTriggers(self, n)

    def SetTrigger(self, timeToNextTrigger, timeToNextTx, syncReq, idx):
        return _iarius.IArius_SetTrigger(self, timeToNextTrigger, timeToNextTx, syncReq, idx)

# Register IArius in _iarius:
_iarius.IArius_swigregister(IArius)


def GetArius(idx):
    return _iarius.GetArius(idx)
class II2CMaster(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr

    def Write(self, address, data, length):
        return _iarius.II2CMaster_Write(self, address, data, length)

    def Read(self, address, data, length):
        return _iarius.II2CMaster_Read(self, address, data, length)

    def WriteAndRead(self, address, writedata, writelength, readdata, readlength):
        return _iarius.II2CMaster_WriteAndRead(self, address, writedata, writelength, readdata, readlength)
    __swig_destroy__ = _iarius.delete_II2CMaster

# Register II2CMaster in _iarius:
_iarius.II2CMaster_swigregister(II2CMaster)


def TransferRXBufferToHostLocation(that, dstAddress, length, srcAddress):
    return _iarius.TransferRXBufferToHostLocation(that, dstAddress, length, srcAddress)

def castToII2CMaster(ptr):
    return _iarius.castToII2CMaster(ptr)


