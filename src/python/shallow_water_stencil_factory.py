from shallow_water_gt4py_config import ShallowWaterGT4PyConfig
import gt4py.gtscript as gtscript
from numpy import float64

# Define float types
F_TYPE = float64
FloatFieldIJ = gtscript.Field[gtscript.IJ, F_TYPE]

# Define constants
g = 9.81

class ShallowWaterStencilFactory:

    def __init__(self, config: ShallowWaterGT4PyConfig):
        self.backend = config.backend

    def makeStencil(self, func):
        return gtscript.stencil(definition=func, backend=self.backend)

    def boundary_update(u     : FloatFieldIJ,
                        v     : FloatFieldIJ,
                        h     : FloatFieldIJ,
                        u_new : FloatFieldIJ,
                        v_new : FloatFieldIJ,
                        h_new : FloatFieldIJ,
                        north : int,
                        south : int,
                        west  : int,
                        east  : int,
                        dtdx  : F_TYPE,
                        dtdy  : F_TYPE):
        # NOTE: FORWARD ordering is required here to disambiguate the missing k dimension
        #       for assignment into our 2D arrays.
        # NOTE: Requires origin/domain set such that I[0], I[-1], J[0], J[-1] are the
        #       first/last elements of the compute domain (which they are in the original
        #       Fortran).
        with computation(FORWARD), interval(...):
            # Update southern boundary if there is one
            with horizontal(region[:, J[0]]):
                if (south == -1):
                    h_new = h[0, 1]
                    u_new = u[0, 1]
                    v_new = -v[0, 1]

            # Update northern boundary if there is one
            with horizontal(region[:, J[-1]]):
                if (north == -1):
                    h_new = h[0,-1]
                    u_new = u[0,-1]
                    v_new = -v[0,-1]

            # Update western boundary if there is one
            with horizontal(region[I[0], :]):
                if (west == -1):
                    h_new = h[1,0]
                    u_new = -u[1,0]
                    v_new = v[1,0]

            # Update eastern boundary if there is one
            with horizontal(region[I[-1], :]):
                if (east == -1):
                    h_new = h[-1,0]
                    u_new = -u[-1,0]
                    v_new = v[-1,0]


    def interior_update(u     : FloatFieldIJ,
                        v     : FloatFieldIJ,
                        h     : FloatFieldIJ,
                        b     : FloatFieldIJ,
                        u_new : FloatFieldIJ,
                        v_new : FloatFieldIJ,
                        h_new : FloatFieldIJ,
                        dtdx  : F_TYPE,
                        dtdy  : F_TYPE):
        # NOTE: FORWARD ordering is required here to disambiguate the missing k dimension
        #       for assignment into our 2D arrays.
        with computation(FORWARD), interval(...):
            u_new = ((u[1,0] + u[-1,0] + u[0,1] + u[0,-1]) / 4.0)           \
                    - 0.5 * dtdx * ((u[1,0]**2) / 2.0 - (u[-1,0]**2) / 2.0) \
                    - 0.5 * dtdy * v * (u[0,1] - u[0,-1])                   \
                    - 0.5 * g * dtdx * (h[1,0] - h[-1,0])

            v_new = ((v[1,0] + v[-1,0] + v[0,1] + v[0,-1]) / 4.0)           \
                    - 0.5 * dtdy * ((v[0,1]**2) / 2.0 - (v[0,1]**2) / 2.0)  \
                    - 0.5 * dtdx * u * (v[1,0] - v[-1,0])                   \
                    - 0.5 * g * dtdy * (h[0,1] - h[0,-1])

            h_new = ((h[1,0] + h[-1,0] + h[0,1] + h[0,-1]) / 4.0)                \
                    - 0.5 * dtdx * u * ((h[1,0] - b[1,0]) - (h[-1,0] - b[-1,0])) \
                    - 0.5 * dtdy * v * ((h[0,1] - b[0,1]) - (h[0,-1] - b[0,-1])) \
                    - 0.5 * dtdx * (h - b) * (u[1,0] - u[-1,0])                  \
                    - 0.5 * dtdy * (h - b) * (v[0,1] - v[0,-1])
