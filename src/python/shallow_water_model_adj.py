from shallow_water_model_config import ShallowWaterModelConfig
from shallow_water_gt4py_config import ShallowWaterGT4PyConfig
from shallow_water_geometry import ShallowWaterGeometry
from shallow_water_state import ShallowWaterState
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage

class ShallowWaterModelADJ:

    def __init__(self, modelConfig : ShallowWaterModelConfig, gt4PyConfig: ShallowWaterGT4PyConfig, geometry : ShallowWaterGeometry):

        self.modelConfig = modelConfig
        self.gt4PyConfig = gt4PyConfig
        self.geometry = geometry
        self.dt = self.modelConfig.dt
        self.backend = self.gt4PyConfig.backend
        self.float_type = self.gt4PyConfig.float_type
        self.field_type = gtscript.Field[gtscript.IJ, self.float_type]

        # Initialize the b array - Currently unused
        self.b = gt_storage.zeros(shape=(self.geometry.xme - self.geometry.xms + 1, self.geometry.yme - self.geometry.yms + 1),
                                  dtype=self.float_type, backend=self.backend, default_origin=(1, 1))

        # Set gravitational acceleration constant
        g = 9.81

        # Define boundary_update stencil function
        def boundary_update(u      : self.field_type,
                            v      : self.field_type,
                            h      : self.field_type,
                            traj_u : self.field_type,
                            traj_v : self.field_type,
                            traj_h : self.field_type,
                            u_new  : self.field_type,
                            v_new  : self.field_type,
                            h_new  : self.field_type,
                            north  : int,
                            south  : int,
                            west   : int,
                            east   : int,
                            dtdx   : self.float_type,
                            dtdy   : self.float_type):
            # NOTE: FORWARD ordering is required here to disambiguate the missing k dimension
            #       for assignment into our 2D arrays.
            # NOTE: Requires origin/domain set such that I[0], I[-1], J[0], J[-1] are the
            #       first/last elements of the compute domain (which they are in the original
            #       Fortran).
            with computation(FORWARD), interval(...):
                # Update eastern boundary if there is one
                with horizontal(region[I[-2], :]):
                    if (east == -1):
                        u = u - u_new[1,0]
                        v = v + v_new[1,0]
                        h = h + h_new[1,0]
                        u_new[1,0] = 0.0
                        v_new[1,0] = 0.0
                        h_new[1,0] = 0.0
                # Update western boundary if there is one
                with horizontal(region[I[1], :]):
                    if (west == -1):
                        u = u - u_new[-1,0]
                        v = v + v_new[-1,0]
                        h = h + h_new[-1,0]
                        u_new[-1,0] = 0.0
                        v_new[-1,0] = 0.0
                        h_new[-1,0] = 0.0
                # Update northern boundary if there is one
                with horizontal(region[:, J[-2]):
                    if (north == -1):
                        u = u + u_new[0,1]
                        v = v - v_new[0,1]
                        h = h + h_new[0,1]
                        u_new = 0.0
                        v_new = 0.0
                        h_new = 0.0
                # Update southern boundary if there is one
                with horizontal(region[:, J[1]]):
                    if (south == -1):
                        u = u + u_new[0,1]
                        v = v - v_new[0,1]
                        h = h + h_new[0,1]


        # Define interior_update stencil function
        def interior_update(u      : self.field_type,
                            v      : self.field_type,
                            h      : self.field_type,
                            traj_u : self.field_type,
                            traj_v : self.field_type,
                            traj_h : self.field_type,
                            b      : self.field_type,
                            u_new  : self.field_type,
                            v_new  : self.field_type,
                            h_new  : self.field_type,
                            dtdx   : self.float_type,
                            dtdy   : self.float_type):
            # NOTE: FORWARD ordering is required here to disambiguate the missing k dimension
            #       for assignment into our 2D arrays.
            with computation(FORWARD), interval(...):
                u_new = (u[1,0] + u[-1,0] + u[0,1] + u[0,-1]) / 4.0                                     \
                        - 0.5 * dtdx * (2.0 * traj_u[1,0] * u[1,0] / 2.0                                \
                        - 2.0 * traj_u[-1,0] * u[-1,0] / 2.0)                                           \
                        - 0.5 * dtdy * (v * (traj_u[0,1] - traj_u[0,-1]) + traj_v * (u[0,1] - u[0,-1])) \
                        - 0.5 * g * dtdx * (h[1,0] - h[-1,0])

                v_new = (v[1,0] + v[-1,0] + v[0,1] + v[0,-1]) / 4.0                                     \
                        - 0.5 * dtdx * (u * (traj_v[1,0] - traj_v[-1,0]) + traj_u * (v[1,0] - v[-1,0])) \
                        - 0.5 * g * dtdy * (h[0,1] - h[0,-1])

                h_new = (h[1,0] + h[-1,0] + h[0,1] + h[0,-1]) / 4.0                  \
                        - 0.5 * dtdx * (u * (traj_h[1,0] - b[1,0] - (traj_h[-1,0]    \
                        - b[-1,0])) + traj_u * (h[1,0] - h[-1,0]))                   \
                        - 0.5 * dtdy * (v * (traj_h[0,1] - b[0,1] - (traj_h[0,-1]    \
                        - b[0,-1])) + traj_v * (h[0,1] - h[0,-1]))                   \
                        - 0.5 * dtdx * (h * (traj_u[1,0] - traj_u[-1,0]) + (traj_h   \
                        - b) * (u[1,0] - u[-1,0]))                                   \
                        - 0.5 * dtdy * (h * (traj_v[0,1] - traj_v[0,-1]) + (traj_h   \
                        - b) * (v[0,1] - v[0,-1]))

        # Compile the stenci functions for the given backend
        self._boundary_update = gtscript.stencil(definition=boundary_update, backend=self.backend)
        self._interior_update = gtscript.stencil(definition=interior_update, backend=self.backend)


    def adv_nsteps(self, state : ShallowWaterState, trajectory: ShallowWaterState, nsteps : int):

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
                                   dtype=self.float_type, backend=self.backend, default_origin=(1, 1))
        _v_new = gt_storage.empty(shape=(self.geometry.xme - self.geometry.xms + 1, self.geometry.yme - self.geometry.yms + 1),
                                   dtype=self.float_type, backend=self.backend, default_origin=(1, 1))
        _h_new = gt_storage.empty(shape=(self.geometry.xme - self.geometry.xms + 1, self.geometry.yme - self.geometry.yms + 1),
                                   dtype=self.float_type, backend=self.backend, default_origin=(1, 1))

        for n in range(nsteps):
            # Exchange halo
            state.exchange_halo()
            trajectory.exchange_halo()

            # Adjoint of update state with new state
            for i in range(self.geometry.xps - self.geometry.xms, self.geometry.xpe - self.geometry.xms + 1):
                for j in range(self.geometry.yps - self.geometry.yms, self.geometry.ype - self.geometry.yms + 1):
                    _h_new[i,j] = state.h[i,j]
                    _v_new[i,j] = state.v[i,j]
                    _u_new[i,j] = state.u[i,j]
                    state.h[i,j] = 0.0
                    state.v[i,j] = 0.0
                    state.u[i,j] = 0.0

            # Get new interior points
            self._interior_update(u=state.u,
                                  v=state.v,
                                  h=state.h,
                                  traj_u=trajectory.u,
                                  traj_v=trajectory.v,
                                  traj_h=trajectory.h,
                                  b=self.b,
                                  u_new=_u_new,
                                  v_new=_v_new,
                                  h_new=_h_new,
                                  dtdx=_dtdx,
                                  dtdy=_dtdy,
                                  origin=(1, 1),
                                  domain=(self.geometry.xte - self.geometry.xts + 1, self.geometry.yte - self.geometry.yts + 1, 1))


            # Get new boundaries
            self._boundary_update(u=state.u,
                                  v=state.v,
                                  h=state.h,
                                  traj_u=trajectory.u,
                                  traj_v=trajectory.v,
                                  traj_h=trajectory.h,
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

            # Update the clock
            state.advance_clock(-self.dt)
