from shallow_water_gt4py_config import ShallowWaterGT4PyConfig
import gt4py.gtscript as gtscript

class ShallowWaterStencilFactory:

    def __init__(self, config: ShallowWaterGT4PyConfig):
        self.backend = config.backend
        self.float_type = config.float_type

    def makeStencil(self, func):
        return gtscript.stencil(definition=func, backend=self.backend)
