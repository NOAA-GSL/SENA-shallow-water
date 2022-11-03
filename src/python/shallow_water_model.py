from shallow_water_model_config import ShallowWaterModelConfig
from shallow_water_geometry import ShallowWaterGeometry
from shallow_water_state import ShallowWaterState
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage

# Define float types
F_TYPE = float
FloatFieldIJ = gtscript.Field[gtscript.IJ, F_TYPE]

# Define constants
g = 9.81

class ShallowWaterModel:

    backend="numpy"



    def __init__(self, config : ShallowWaterModelConfig, geometry : ShallowWaterGeometry):

        self.config = config
        self.geometry = geometry
        self.dt = self.config.dt

        self.b = gt_storage.zeros(shape=(self.geometry.xme - self.geometry.xms + 1, self.geometry.yme - self.geometry.yms + 1),
                                  dtype=F_TYPE, backend=self.backend, default_origin=(1, 1))


    def adv_nsteps(self, state : ShallowWaterState, nsteps : int):

        # Get grid spacing
        _dx = self.geometry.dx
        _dy = self.geometry.dy

        # Sanity check for timestep
        if (state.max_wavespeed > 0.0):
            _maxdt = 0.68 * min(_dx, _dy) / state.max_wavespeed
            if (self.dt > _maxdt):
                print(f"WARNING: time step is too large, should be <= {_maxdt}")

        # Compute dxdt and dydt
        _dtdx = self.dt / _dx
        _dtdy = self.dt / _dy

        # Create gt4py storages for the new model state
        _u_new = gt_storage.empty(shape=(self.geometry.xme - self.geometry.xms + 1, self.geometry.yme - self.geometry.yms + 1),
                                   dtype=F_TYPE, backend=self.backend, default_origin=(1, 1))
        _v_new = gt_storage.empty(shape=(self.geometry.xme - self.geometry.xms + 1, self.geometry.yme - self.geometry.yms + 1),
                                   dtype=F_TYPE, backend=self.backend, default_origin=(1, 1))
        _h_new = gt_storage.empty(shape=(self.geometry.xme - self.geometry.xms + 1, self.geometry.yme - self.geometry.yms + 1),
                                   dtype=F_TYPE, backend=self.backend, default_origin=(1, 1))

        for n in range(nsteps):
            # Exchange halo
            state.exchange_halo()

            # Get new boundaries
            self._boundary_update(u=state.u,
                                  v=state.v,
                                  h=state.h,
                                  u_new=_u_new,
                                  v_new=_v_new,
                                  h_new=_h_new,
                                  north=self.geometry.north,
                                  south=self.geometry.south,
                                  west=self.geometry.west,
                                  east=self.geometry.east,
                                  dtdx=_dtdx,
                                  dtdy=_dtdy,
                                  origin=(0, 0),
                                  domain=(self.geometry.xme - self.geometry.xms + 1, self.geometry.yme - self.geometry.yms + 1, 1))

            # Get new interior points
            self._interior_update(u=state.u,
                                  v=state.v,
                                  h=state.h,
                                  b=self.b,
                                  u_new=_u_new,
                                  v_new=_v_new,
                                  h_new=_h_new,
                                  dtdx=_dtdx,
                                  dtdy=_dtdy,
                                  origin=(1, 1),
                                  domain=(self.geometry.xte - self.geometry.xts + 1, self.geometry.yte - self.geometry.yts + 1, 1))

            # Update state with new state
            for i in range(self.geometry.xps - self.geometry.xms, self.geometry.xpe - self.geometry.xms + 1):
                for j in range(self.geometry.yps - self.geometry.yms, self.geometry.ype - self.geometry.yms + 1):
                    state.u[i,j] = _u_new[i,j]
                    state.v[i,j] = _v_new[i,j]
                    state.h[i,j] = _h_new[i,j]

            # Update the clock
            state.advance_clock(self.dt)

    @gtscript.stencil(backend=backend)
    def _boundary_update(u     : FloatFieldIJ,
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


    @gtscript.stencil(backend=backend)
    def _interior_update(u     : FloatFieldIJ,
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
