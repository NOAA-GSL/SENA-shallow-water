import dataclasses
import gt4py
import yaml
import numpy as np
import dacite
import gt4py.gtscript as gtscript
from mpi4py import MPI

if MPI is not None:
    import os

    gt4py.config.cache_settings["dir_name"] = os.environ.get(
        "GT_CACHE_DIR_NAME", f".gt_cache_{MPI.COMM_WORLD.Get_rank():06}"
    )

from gt4py.gtscript import Field

F_TYPE = np.float64
I_TYPE = np.int32
# Use a 2D Field for the GT4Py Stencils
FloatFieldIJ = gtscript.Field[gtscript.IJ, np.float64]

@dataclasses.dataclass
class StencilConfig:
    gt4py: dict
    runtime: dict
    geometry: dict
    model: dict

    def compile_to_stencil(self, defn_func):
        return gtscript.stencil(
            definition=defn_func,
            backend=self.gt4py['backend']
        )

# Get model state one step in the future for the domain interior
# @gtscript.stencil(backend=backend)
def update_interior(u:     FloatFieldIJ,
                    v:     FloatFieldIJ,
                    h:     FloatFieldIJ,
                    b:     FloatFieldIJ,
                    u_new: FloatFieldIJ,
                    v_new: FloatFieldIJ,
                    h_new: FloatFieldIJ,
                    dtdx:  F_TYPE,
                    dtdy:  F_TYPE,
                    g:     F_TYPE):

        with computation(FORWARD), interval(...):
            u_new = ((u[1,0] + u[-1,0] + u[0,1] + u[0,-1]) / 4.0)                    \
                            - 0.5 * dtdx * ((u[1,0]**2) / 2.0 - (u[-1,0]**2) / 2.0)  \
                            - 0.5 * dtdy * (v[0,0]) * (u[0,1] - u[0,-1])             \
                            - 0.5 * g * dtdx * (h[1,0] - h[-1,0])
           
            v_new = ((v[1,0] + v[-1,0] + v[0,1] + v[0,-1]) / 4.0)                    \
                            - 0.5 * dtdy * ((v[0,1]**2) / 2.0 - (v[0,1]**2) / 2.0)   \
                            - 0.5 * dtdx * (u[0,0]) * (v[1,0] - v[-1,0])             \
                            - 0.5 * g * dtdy * (h[0,1] - h[0,-1])
           
            h_new = ((h[1,0] + h[-1,0] + h[0,1] + h[0,-1]) / 4.0)                                \
                            - 0.5 * dtdx * (u[0,0]) * ((h[1,0] - b[1,0]) - (h[-1,0] - b[-1,0]))  \
                            - 0.5 * dtdy * (v[0,0]) * ((h[0,1] - b[0,1]) - (h[0,-1] - b[0,-1]))  \
                            - 0.5 * dtdx * (h[0,0] - b[0,0]) * (u[1,0] - u[-1,0])                \
                            - 0.5 * dtdy * (h[0,0] - b[0,0]) * (v[0,1] - v[0,-1])

# Get model state one step in the future for the domain boundaries
# @gtscript.stencil(backend=backend)
def update_boundaries(north :  int,
                      south :  int,
                      east  :  int,
                      west  :  int,
                      u     :  FloatFieldIJ,
                      v     :  FloatFieldIJ,
                      h     :  FloatFieldIJ,
                      u_new :  FloatFieldIJ,
                      v_new :  FloatFieldIJ,
                      h_new :  FloatFieldIJ):
    
    with computation(FORWARD), interval(...):
        
        # Update southern boundary
        with horizontal(region[:, J[0]]):
            if (south == -1): 
                h_new =  h[0,1]
                u_new =  u[0,1]
                v_new = -v[0,1]
        
        # Update northern boundary
        with horizontal(region[:, J[-1]]):
            if (north == -1):
                h_new =  h[0,-1] 
                u_new =  u[0,-1] 
                v_new = -v[0,-1] 
        
        # Update western boundary
        with horizontal(region[I[0], :]):
            if (west == -1):
                h_new =  h[1,0]
                u_new = -u[1,0]
                v_new =  v[1,0]
        
        # Update eastern boundary
        with horizontal(region[I[-1], :]):
            if (east == -1):
                h_new =  h[-1,0]
                u_new = -u[-1,0]
                v_new =  v[-1,0]

def update_interior_tl(traj_u:     FloatFieldIJ,
                       traj_v:     FloatFieldIJ,
                       traj_h:     FloatFieldIJ,
                       u:          FloatFieldIJ,
                       v:          FloatFieldIJ,
                       h:          FloatFieldIJ,
                       b:          FloatFieldIJ,
                       u_new:      FloatFieldIJ,
                       v_new:      FloatFieldIJ,
                       h_new:      FloatFieldIJ,
                       dtdx:       F_TYPE,
                       dtdy:       F_TYPE,
                       g:          F_TYPE):
        
        # Employ lax
        with computation(FORWARD), interval(...):
            u_new = (u[1,0] + u[-1,0] + u[0,1] + u[0,-1]) / 4.0                                                         \
                            - 0.5 * dtdx * (2 * traj_u[1,0] * u[1,0] / 2.0                                              \
                            - 2.0 * traj_u[-1,0] * u[-1, 0] / 2.0)                                                      \
                            - 0.5 * dtdy * (v[0,0] * (traj_u[0,1] - traj_u[0,-1]) + traj_v[0,0] * (u[0,1] - u[0,-1]))   \
                            - 0.5 * g * dtdx * (h[1,0] - h[-1,0])
            
            v_new = (v[1,0] + v[-1,0] + v[0,1] + v[0,-1]) / 4.0                                                         \
                            - 0.5 * dtdx * (u[0,0] * (traj_v[1,0] - traj_v[-1,0]) + traj_u[0,0] * (v[1,0] - v[-1,0]))   \
                            - 0.5 * g * dtdy * (h[0,1] - h[0,-1])
            
            h_new = (h[1,0] + h[-1,0] + h[0,1] + h[0,-1]) / 4.0                                                         \
                            - 0.5 * dtdx * (u[0,0] * (traj_h[1,0] - b[1,0] - (traj_h[-1,0]                              \
                            - b[-1,0])) + traj_u[0,0] * (h[1,0] - h[-1,0]))                                             \
                            - 0.5 * dtdy * (v[0,0] * (traj_h[0,1] - b[0,1] - (traj_h[0,-1]                              \
                            - b[0,-1])) + traj_v[0,0] * (h[0,1] - h[0,-1]))                                             \
                            - 0.5 * dtdx * (h[0,0] * (traj_u[1,0] - traj_u[-1,0]) + (traj_h[0,0]                        \
                            - b[0,0]) * (u[1,0] - u[-1,0]))                                                             \
                            - 0.5 * dtdy * (h[0,0] * (traj_v[0,1] - traj_v[0,-1]) + (traj_h[0,0]                        \
                            - b[0,0]) * (v[0,1] - v[0,-1]))


def update_interior_adj():
    pass

def update_boundaries_adj():
    pass

class StencilFunctions:

    def __init__(self, config: StencilConfig):
        self.update_interior = config.compile_to_stencil(update_interior)
        self.update_boundaries = config.compile_to_stencil(update_boundaries)
        self.update_interior_tl = config.compile_to_stencil(update_interior_tl)        
        # self.update_interior_adj = config.compile_to_stencil(update_interior_adj)
        # self.update_boundaries_adj = config.compile_to_stencil(update_boundaries_adj)        

if __name__ == "StencilFactory":
    config = dacite.from_dict(
        data_class=StencilConfig,
        data=yaml.safe_load(open('../../parm/shallow_water.yml')),
        config=dacite.Config(strict=True)
    )
    stencils = StencilFunctions(config)

if __name__ == '__main__':
    config = dacite.from_dict(
        data_class=StencilConfig,
        data=yaml.safe_load(open('../../parm/shallow_water.yml')),
        config=dacite.Config(strict=True)
    )
    stencils = StencilFunctions(config)
    
