import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage

from shallow_water_model_config import ShallowWaterModelConfig
from shallow_water_gt4py_config import ShallowWaterGT4PyConfig
from shallow_water_geometry import ShallowWaterGeometry
from shallow_water_state import ShallowWaterState

class ShallowWaterModelADJ:

    def __init__(self, modelConfig : ShallowWaterModelConfig, gt4PyConfig: ShallowWaterGT4PyConfig, geometry : ShallowWaterGeometry):

        self.modelConfig = modelConfig
        self.gt4PyConfig = gt4PyConfig
        self.geometry = geometry
        self.dt = self.modelConfig.dt
        self.backend = self.gt4PyConfig.backend
        self.float_type = self.gt4PyConfig.float_type
        self.field_type = gtscript.Field[gtscript.IJ, self.float_type]

        # Set gravitational acceleration constant
        g = 9.81

        # Initialize the b array - Currently unused
        self.b = gt_storage.zeros(shape=(self.geometry.xme - self.geometry.xms + 1, self.geometry.yme - self.geometry.yms + 1),
                                  dtype=self.float_type, backend=self.backend, default_origin=(1, 1))

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

            # Update eastern boundary if there is one
            with computation(FORWARD), interval(...):
                with horizontal(region[I[-2], :]):
                    if (east == -1):
                        u = u - u_new[1,0]
                        v = v + v_new[1,0]
                        h = h + h_new[1,0]
            with computation(FORWARD), interval(...):
                with horizontal(region[I[-1], :]):
                    if (east == -1):
                        u_new = 0.0
                        v_new = 0.0
                        h_new = 0.0

            # Update western boundary if there is one
            with computation(FORWARD), interval(...):
               with horizontal(region[I[1], :]):
                   if (west == -1):
                       u = u - u_new[-1,0]
                       v = v + v_new[-1,0]
                       h = h + h_new[-1,0]
            with computation(FORWARD), interval(...):
                with horizontal(region[I[0], :]):
                    if (west == -1):
                        u_new = 0.0
                        v_new = 0.0
                        h_new = 0.0

            # Update northern boundary if there is one
            with computation(FORWARD), interval(...):
                with horizontal(region[:, J[-2]]):
                    if (north == -1):
                        u = u + u_new[0,1]
                        v = v - v_new[0,1]
                        h = h + h_new[0,1]
            with computation(FORWARD), interval(...):
                with horizontal(region[:, J[-1]]):
                    if (north == -1):
                        u_new = 0.0
                        v_new = 0.0
                        h_new = 0.0

            # Update southern boundary if there is one
            with computation(FORWARD), interval(...):
                with horizontal(region[:, J[1]]):
                    if (south == -1):
                        u = u + u_new[0,-1]
                        v = v - v_new[0,-1]
                        h = h + h_new[0,-1]

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
                            north  : int,
                            south  : int,
                            west   : int,
                            east   : int,
                            dtdx   : self.float_type,
                            dtdy   : self.float_type):
            # NOTE: FORWARD ordering is required here to disambiguate the missing k dimension
            #       for assignment into our 2D arrays.

            from __externals__ import g

            with computation(FORWARD), interval(...):

                # Take care of our northern neighbor's southernmost j-1
                with horizontal(region[I[1]:I[-1], J[-2]]):
                    if (north != -1):
                        tempb = h_new[0,1] / 4.0
                        tempb2 = -(dtdy * 0.5 * h_new[0,1])
                        tempb3 = traj_v[0,1] * tempb2
                        tempb6 = -(dtdy * 0.5 * h_new[0,1])
                        tempb7 = (traj_h[0,1] - b[0,1]) * tempb6
                        tempb8 = v_new[0,1] / 4.0
                        tempb11 = -(g * 0.5 * dtdy * v_new[0,1])
                        tempb12 = u_new[0,1] / 4.0
                        tempb14 = -(dtdy * 0.5 * u_new[0,1])
                        tempb15 = traj_v[0,1] * tempb14
                        u = u + tempb12 - tempb15
                        v = v - tempb7 + tempb8
                        h = h + tempb - tempb3 - tempb11

                # Take care of our interior j

                # Take care of our eastern neighbor's westernmost i-1
                with horizontal(region[I[-2], J[1]:J[-1]]):
                    if (east != -1):
                        tempb = h_new[1,0] / 4.0
                        tempb0 = -(dtdx * 0.5 * h_new[1,0])
                        tempb1 = traj_u[1,0] * tempb0
                        tempb4 = -(dtdx * 0.5 * h_new[1,0])
                        tempb5 = (traj_h[1,0] - b[1,0]) * tempb4
                        tempb8 = v_new[1,0] / 4.0
                        tempb9 = -(dtdx * 0.5 * v_new[1,0])
                        tempb10 = traj_u[1,0] * tempb9
                        tempb12 = u_new[1,0] / 4.0
                        tempb13 = -(dtdx * 0.5 * u_new[1,0])
                        tempb16 = -(g * 0.5 * dtdx * u_new[1,0])
                        u = u - tempb5 + tempb12 - 2.0 * traj_u * tempb13 / 2.0
                        v = v + tempb8 - tempb10
                        h = h + tempb - tempb1 - tempb16

                # Take care of our interior i #1
                with horizontal(region[I[0]:I[-1]-1, J[1]:J[-1]]):
                    tempb = h_new[1,0] / 4.0
                    tempb0 = -(dtdx * 0.5 * h_new[1,0])
                    tempb1 = traj_u[1,0] * tempb0
                    tempb2 = -(dtdy * 0.5 * h_new[1,0])
                    tempb3 = traj_v[1,0] * tempb2
                    tempb4 = -(dtdx * 0.5 * h_new[1,0])
                    tempb5 = (traj_h[1,0] - b[1,0]) * tempb4
                    tempb6 = -(dtdy * 0.5 * h_new[1,0])
                    tempb7 = (traj_h[1,0] - b[1,0]) * tempb6
                    tempb8 = v_new[1,0] / 4.0
                    tempb9 = -(dtdx * 0.5 * v_new[1,0])
                    tempb10 = traj_u[1,0] * tempb9
                    tempb11 = -(g * 0.5 * dtdy * v_new[1,0])
                    tempb12 = u_new[1,0] / 4.0
                    tempb13 = -(dtdx * 0.5 * u_new[1,0])
                    tempb14 = -(dtdy * 0.5 * u_new[1,0])
                    tempb15 = traj_v[1,0] * tempb14
                    tempb16 = -(g * 0.5 * dtdx * u_new[1,0])
                    u = u - tempb5 + tempb12 - 2.0 * traj_u * tempb13 / 2.0
                    v = v + tempb8 - tempb10
                    h = h + tempb - tempb1 - tempb16

                # Take care of our interior i #2
                with horizontal(region[I[1]:I[-1], J[1]:J[-1]]):
                    tempb = h_new / 4.0
                    tempb0 = -(dtdx * 0.5 * h_new)
                    tempb1 = traj_u * tempb0
                    tempb2 = -(dtdy * 0.5 * h_new)
                    tempb3 = traj_v * tempb2
                    tempb4 = -(dtdx * 0.5 * h_new)
                    tempb5 = (traj_h - b) * tempb4
                    tempb6 = -(dtdy * 0.5 * h_new)
                    tempb7 = (traj_h - b) * tempb6
                    tempb8 = v_new / 4.0
                    tempb9 = -(dtdx * 0.5 * v_new)
                    tempb10 = traj_u * tempb9
                    tempb11 = -(g * 0.5 * dtdy * v_new)
                    tempb12 = u_new / 4.0
                    tempb13 = -(dtdx * 0.5 * u_new)
                    tempb14 = -(dtdy * 0.5 * u_new)
                    tempb15 = traj_v * tempb14
                    tempb16 = -(g * 0.5 * dtdx * u_new)
                    u = u + (b[-1,0] - b[1,0] + traj_h[1,0] - traj_h[-1,0]) * tempb0 + (traj_v[1,0] - traj_v[-1,0]) * tempb9
                    v = v + (b[0,-1] - b[0,1] + traj_h[0,1] - traj_h[0,-1]) * tempb2 + (traj_u[0,1] - traj_u[0,-1]) * tempb14
                    h = h + (traj_v[0,1] - traj_v[0,-1]) * tempb6 + (traj_u[1,0] - traj_u[-1,0]) * tempb4

                # Take care of our interior i #3
                with horizontal(region[I[1]+1:I[-1]+1, J[1]:J[-1]]):
                    tempb = h_new[-1,0] / 4.0
                    tempb0 = -(dtdx * 0.5 * h_new[-1,0])
                    tempb1 = traj_u[-1,0] * tempb0
                    tempb2 = -(dtdy * 0.5 * h_new[-1,0])
                    tempb3 = traj_v[-1,0] * tempb2
                    tempb4 = -(dtdx * 0.5 * h_new[-1,0])
                    tempb5 = (traj_h[-1,0] - b[-1,0]) * tempb4
                    tempb6 = -(dtdy * 0.5 * h_new[-1,0])
                    tempb7 = (traj_h[-1,0] - b[-1,0]) * tempb6
                    tempb8 = v_new[-1,0] / 4.0
                    tempb9 = -(dtdx * 0.5 * v_new[-1,0])
                    tempb10 = traj_u[-1,0] * tempb9
                    tempb11 = -(g * 0.5 * dtdy * v_new[-1,0])
                    tempb12 = u_new[-1,0] / 4.0
                    tempb13 = -(dtdx * 0.5 * u_new[-1,0])
                    tempb14 = -(dtdy * 0.5 * u_new[-1,0])
                    tempb15 = traj_v[-1,0] * tempb14
                    tempb16 = -(g * 0.5 * dtdx * u_new[-1,0])
                    u = u + tempb5 + 2.0 * traj_u * tempb13 / 2.0 + tempb12
                    v = v + tempb10 + tempb8
                    h = h + tempb1 + tempb + tempb16

                # Take care of our interior i #4
                with horizontal(region[I[1]:I[-1], J[0]:J[-1]-1]):
                    tempb = h_new[0,1] / 4.0
                    tempb0 = -(dtdx * 0.5 * h_new[0,1])
                    tempb1 = traj_u[0,1] * tempb0
                    tempb2 = -(dtdy * 0.5 * h_new[0,1])
                    tempb3 = traj_v[0,1] * tempb2
                    tempb4 = -(dtdx * 0.5 * h_new[0,1])
                    tempb5 = (traj_h[0,1] - b[0,1]) * tempb4
                    tempb6 = -(dtdy * 0.5 * h_new[0,1])
                    tempb7 = (traj_h[0,1] - b[0,1]) * tempb6
                    tempb8 = v_new[0,1] / 4.0
                    tempb9 = -(dtdx * 0.5 * v_new[0,1])
                    tempb10 = traj_u[0,1] * tempb9
                    tempb11 = -(g * 0.5 * dtdy * v_new[0,1])
                    tempb12 = u_new[0,1] / 4.0
                    tempb13 = -(dtdx * 0.5 * u_new[0,1])
                    tempb14 = -(dtdy * 0.5 * u_new[0,1])
                    tempb15 = traj_v[0,1] * tempb14
                    tempb16 = -(g * 0.5 * dtdx * u_new[0,1])
                    u = u + tempb12 - tempb15
                    v = v - tempb7 + tempb8
                    h = h + tempb - tempb3 - tempb11

                # Take care of our interior i #5
                with horizontal(region[I[1]:I[-1], J[2]:J[-1]+1]):
                    tempb = h_new[0,-1] / 4.0
                    tempb0 = -(dtdx * 0.5 * h_new[0,-1])
                    tempb1 = traj_u[0,-1] * tempb0
                    tempb2 = -(dtdy * 0.5 * h_new[0,-1])
                    tempb3 = traj_v[0,-1] * tempb2
                    tempb4 = -(dtdx * 0.5 * h_new[0,-1])
                    tempb5 = (traj_h[0,-1] - b[0,-1]) * tempb4
                    tempb6 = -(dtdy * 0.5 * h_new[0,-1])
                    tempb7 = (traj_h[0,-1] - b[0,-1]) * tempb6
                    tempb8 = v_new[0,-1] / 4.0
                    tempb9 = -(dtdx * 0.5 * v_new[0,-1])
                    tempb10 = traj_u[0,-1] * tempb9
                    tempb11 = -(g * 0.5 * dtdy * v_new[0,-1])
                    tempb12 = u_new[0,-1] / 4.0
                    tempb13 = -(dtdx * 0.5 * u_new[0,-1])
                    tempb14 = -(dtdy * 0.5 * u_new[0,-1])
                    tempb15 = traj_v[0,-1] * tempb14
                    tempb16 = -(g * 0.5 * dtdx * u_new[0,-1])
                    u = u + tempb15 + tempb12
                    v = v + tempb7 + tempb8
                    h = h + tempb3 + tempb + tempb11

                # Take care of our western neighbor's easternmost i+1
                with horizontal(region[I[1], J[1]:J[-1]]):
                    if (west != -1):
                        tempb = h_new[-1,0] / 4.0
                        tempb0 = -(dtdx * 0.5 * h_new[-1,0])
                        tempb1 = traj_u[-1,0] * tempb0
                        tempb4 = -(dtdx * 0.5 * h_new[-1,0])
                        tempb5 = (traj_h[-1,0] - b[-1,0]) * tempb4
                        tempb8 = v_new[-1,0] / 4.0
                        tempb9 = -(dtdx * 0.5 * v_new[-1,0])
                        tempb10 = traj_u[-1,0] * tempb9
                        tempb12 = u_new[-1,0] / 4.0
                        tempb13 = -(dtdx * 0.5 * u_new[-1,0])
                        tempb16 = -(g * 0.5 * dtdx * u_new[-1,0])
                        u = u + tempb5 + 2.0 * traj_u * tempb13 / 2.0 + tempb12
                        v = v + tempb10 + tempb8
                        h = h + tempb1 + tempb + tempb16

                # Take care of our sourthern neighbor's northernmost j+1
                with horizontal(region[I[1]:I[-1], J[1]]):
                    if (south != -1):
                        tempb = h_new[0,-1] / 4.0
                        tempb2 = -(dtdy * 0.5 * h_new[0,-1])
                        tempb3 = traj_v[0,-1] * tempb2
                        tempb6 = -(dtdy * 0.5 * h_new[0,-1])
                        tempb7 = (traj_h[0,-1] - b[0,-1]) * tempb6
                        tempb8 = v_new[0,-1] / 4.0
                        tempb11 = -(g * 0.5 * dtdy * v_new[0,-1])
                        tempb12 = u_new[0,-1] / 4.0
                        tempb14 = -(dtdy * 0.5 * u_new[0,-1])
                        tempb15 = traj_v[0,-1] * tempb14
                        u = u + tempb15 + tempb12
                        v = v + tempb7 + tempb8
                        h = h + tempb3 + tempb + tempb11

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
                u_new = u
                v_new = v
                h_new = h
                u = 0.0
                v = 0.0
                h = 0.0

        # Compile the stenci functions for the given backend
        opts: Dict[str, Any] = {"externals": {"g": self.g}}
        self._boundary_update = gtscript.stencil(definition=boundary_update, backend=self.backend)
        self._interior_update = gtscript.stencil(definition=interior_update, backend=self.backend, **opts)
        self._copy_update = gtscript.stencil(definition=copy_update, backend=self.backend)


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

        # Get local bounds exluding the halo
        _xps = self.geometry.xps
        _xpe = self.geometry.xpe
        _yps = self.geometry.yps
        _ype = self.geometry.ype

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
            trajectory.exchange_halo()

            # Adjoint of update state with new state
            self._copy_update(u = state.u,
                              v = state.v,
                              h = state.h,
                              u_new = _u_new,
                              v_new = _v_new,
                              h_new = _h_new,
                              origin=(0, 0),
                              domain=(_xme - _xms + 1, _yme - _yms + 1, 1))

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
                                  north=self.geometry.north,
                                  south=self.geometry.south,
                                  west=self.geometry.west,
                                  east=self.geometry.east,
                                  dtdx=_dtdx,
                                  dtdy=_dtdy,
                                  origin=(0, 0),
                                  domain=(_xme - _xms + 1, _yme - _yms + 1, 1))

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
                                  domain=(_xme - _xms + 1, _yme - _yms + 1, 1))

            # Update the clock
            state.advance_clock(-self.dt)

