from shallow_water_model_config import ShallowWaterModelConfig
from shallow_water_gt4py_config import ShallowWaterGT4PyConfig
from shallow_water_geometry import ShallowWaterGeometry
from shallow_water_state import ShallowWaterState
from shallow_water_stencil_factory import ShallowWaterStencilFactory
import gt4py.storage as gt_storage
from numpy import float64

# Define float types
F_TYPE = float64

class ShallowWaterModel:

    def __init__(self, modelConfig : ShallowWaterModelConfig, gt4PyConfig: ShallowWaterGT4PyConfig, geometry : ShallowWaterGeometry):

        self.modelConfig = modelConfig
        self.gt4PyConfig = gt4PyConfig
        self.geometry = geometry
        self.dt = self.modelConfig.dt
        self.backend = self.gt4PyConfig.backend

        self.b = gt_storage.zeros(shape=(self.geometry.xme - self.geometry.xms + 1, self.geometry.yme - self.geometry.yms + 1),
                                  dtype=F_TYPE, backend=self.backend, default_origin=(1, 1))
        _stencilFactory = ShallowWaterStencilFactory(self.gt4PyConfig)
        self._boundary_update = _stencilFactory.makeStencil(ShallowWaterStencilFactory.boundary_update)
        self._interior_update = _stencilFactory.makeStencil(ShallowWaterStencilFactory.interior_update)

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
