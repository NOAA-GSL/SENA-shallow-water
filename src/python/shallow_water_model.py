import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage

from shallow_water_model_config import ShallowWaterModelConfig
from shallow_water_gt4py_config import ShallowWaterGT4PyConfig
from shallow_water_geometry import ShallowWaterGeometry
from shallow_water_state import ShallowWaterState

class ShallowWaterModel:

    def __init__(self, modelConfig : ShallowWaterModelConfig, gt4PyConfig: ShallowWaterGT4PyConfig, geometry : ShallowWaterGeometry):

        self.modelConfig = modelConfig
        self.gt4PyConfig = gt4PyConfig
        self.geometry = geometry
        self.dt = self.modelConfig.dt
        self.backend = self.gt4PyConfig.backend
        self.float_type = self.gt4PyConfig.float_type
        self.field_type = gtscript.Field[gtscript.IJ, self.float_type]

        # Set gravitational acceleration constant
        self.g = 9.81

        # Initialize the b array - Currently unused
        self.b = gt_storage.zeros(shape=(self.geometry.xme - self.geometry.xms + 1, self.geometry.yme - self.geometry.yms + 1),
                                  dtype=self.float_type, backend=self.backend, default_origin=(1, 1))


        # Define boundary_update stencil function
        def boundary_update(u     : self.field_type,
                            v     : self.field_type,
                            h     : self.field_type,
                            u_new : self.field_type,
                            v_new : self.field_type,
                            h_new : self.field_type,
                            north : int,
                            south : int,
                            west  : int,
                            east  : int,
                            dtdx  : self.float_type,
                            dtdy  : self.float_type):
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

        # Define interior_update stencil function
        def interior_update(u     : self.field_type,
                            v     : self.field_type,
                            h     : self.field_type,
                            b     : self.field_type,
                            u_new : self.field_type,
                            v_new : self.field_type,
                            h_new : self.field_type,
                            dtdx  : self.float_type,
                            dtdy  : self.float_type):
            # NOTE: FORWARD ordering is required here to disambiguate the missing k dimension
            #       for assignment into our 2D arrays.

            from __externals__ import g

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

        # Define state copy update stencil function
        def copy_update(u     : self.field_type,
                        v     : self.field_type,
                        h     : self.field_type,
                        u_new : self.field_type,
                        v_new : self.field_type,
                        h_new : self.field_type):
            # NOTE: FORWARD ordering is required here to disambiguate the missing k dimension
            #       for assignment into our 2D arrays.
            with computation(FORWARD), interval(...):
                u = u_new
                v = v_new
                h = h_new

        # Compile the stencil functions for the given backend
        opts: Dict[str, Any] = {"externals": {"g": self.g}}
        self._boundary_update = gtscript.stencil(definition=boundary_update, backend=self.backend)
        self._interior_update = gtscript.stencil(definition=interior_update, backend=self.backend, **opts)
        self._copy_update = gtscript.stencil(definition=copy_update, backend=self.backend)


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

        # Get local bounds exluding the halo
        _xps = self.geometry.xps
        _xpe = self.geometry.xpe
        _yps = self.geometry.yps
        _ype = self.geometry.ype

        # Get local interior points 
        _xts = self.geometry.xts
        _xte = self.geometry.xte
        _yts = self.geometry.yts
        _yte = self.geometry.yte

        # Get local bounds including halo
        _xms = self.geometry.xms
        _xme = self.geometry.xme
        _yms = self.geometry.yms
        _yme = self.geometry.yme

        # Create gt4py storages for the new model state
        _u_new = gt_storage.empty(shape=(_xme - _xms + 1, _yme - _yms + 1),
                                  dtype=self.float_type, backend=self.backend, default_origin=(1, 1))
        _v_new = gt_storage.empty(shape=(_xme - _xms + 1, _yme - _yms + 1),
                                  dtype=self.float_type, backend=self.backend, default_origin=(1, 1))
        _h_new = gt_storage.empty(shape=(_xme - _xms + 1, _yme - _yms + 1),
                                  dtype=self.float_type, backend=self.backend, default_origin=(1, 1))

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
                                  domain=(_xme - _xms + 1, _yme - _yms + 1, 1),
                                  validate_args=False)

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
                                  domain=(_xte - _xts + 1, _yte - _yts + 1, 1),
                                  validate_args=False)

            # Update state with new state
            self._copy_update(u = state.u,
                              v = state.v,
                              h = state.h,
                              u_new = _u_new,
                              v_new = _v_new,
                              h_new = _h_new,
                              origin=(_xps - _xms, _yps - _yms),
                              domain=(_xpe - _xps + 1, _ype - _yps + 1, 1),
                              validate_args=False)

            # Update the clock
            state.advance_clock(self.dt)
