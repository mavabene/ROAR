from pydantic import BaseModel, Field
from ROAR.control_module.controller import Controller
from ROAR.utilities_module.vehicle_models import VehicleControl, Vehicle

from ROAR.utilities_module.data_structures_models import Transform, Location
from collections import deque
import numpy as np
import math
import logging
from ROAR.agent_module.agent import Agent
from typing import Tuple
import json
from pathlib import Path


class PIDRollController(Controller):
    def __init__(self, agent, steering_boundary: Tuple[float, float],
                 throttle_boundary: Tuple[float, float], **kwargs):
        super().__init__(agent, **kwargs)
        self.max_speed = math.ceil(2*self.agent.agent_settings.max_speed)
        self.throttle_boundary = throttle_boundary
        self.steering_boundary = steering_boundary
        self.config = json.load(Path(agent.agent_settings.pid_config_file_path).open(mode='r'))
        self.long_pid_controller = LongPIDController(agent=agent,
                                                     throttle_boundary=throttle_boundary,
                                                     max_speed=self.max_speed,
                                                     config=self.config["longitudinal_controller"])
        self.lat_pid_controller = LatPIDController(
            agent=agent,
            config=self.config["latitudinal_controller"],
            steering_boundary=steering_boundary
        )
        self.logger = logging.getLogger(__name__)

    def run_in_series(self, next_waypoint: Transform, **kwargs) -> VehicleControl:
        throttle = self.long_pid_controller.run_in_series(next_waypoint=next_waypoint,
                                                          target_speed=kwargs.get("target_speed", self.max_speed))
        steering = self.lat_pid_controller.run_in_series(next_waypoint=next_waypoint)
        return VehicleControl(throttle=throttle, steering=steering)

    @staticmethod
    def find_k_values(vehicle: Vehicle, config: dict) -> np.array:
        current_speed = Vehicle.get_speed(vehicle=vehicle)
        k_p, k_d, k_i = .5, 0.1, 0
        for speed_upper_bound, kvalues in config.items():
            speed_upper_bound = float(speed_upper_bound)
            if current_speed < speed_upper_bound:
                k_p, k_d, k_i = kvalues["Kp"]*.4, kvalues["Kd"]*.3, kvalues["Ki"]*.05 #******* lowered gain for smoothness
                break
        return np.clip([k_p, k_d, k_i], a_min=0, a_max=1)


# *** original Roll ContRoller + v2 ***
class LongPIDController(Controller):
    def __init__(self, agent, config: dict, throttle_boundary: Tuple[float, float], max_speed: float,
                 dt: float = 0.03, **kwargs):
        super().__init__(agent, **kwargs)
        self.config = config
        self.max_speed = max_speed
        self.throttle_boundary = throttle_boundary
        self._error_buffer = deque(maxlen=10)

        self._dt = dt

    def run_in_series(self, next_waypoint: Transform, **kwargs) -> float:
        target_speed = min(self.max_speed, kwargs.get("target_speed", self.max_speed))
        # self.logger.debug(f"Target_Speed: {target_speed} | max_speed = {self.max_speed}")
        current_speed = Vehicle.get_speed(self.agent.vehicle)

        print('max speed: ',self.max_speed)

        k_p, k_d, k_i = PIDRollController.find_k_values(vehicle=self.agent.vehicle, config=self.config)
        error = target_speed - current_speed

        self._error_buffer.append(error)


        #****************** implement look ahead *******************
        la_err = self.la_calcs(next_waypoint)
        # kla = .09
        #kla = 1/11000 # *** calculated ***
        kla = 1/10000 # *** tuned ***

        if len(self._error_buffer) >= 2:
            # print(self._error_buffer[-1], self._error_buffer[-2])
            _de = (self._error_buffer[-2] - self._error_buffer[-1]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0
        # output = float(np.clip((k_p * error) + (k_d * _de) + (k_i * _ie), self.throttle_boundary[0],
        #                        self.throttle_boundary[1]))
        print(self.agent.vehicle.transform.rotation.roll)
        vehroll = self.agent.vehicle.transform.rotation.roll
        if current_speed >= (target_speed + 2):  # *** reduces speed at max limit more smoothly
            out = 1 - .08 * (current_speed - target_speed)
        # *** old guesses ***
        # else:
        #     if abs(self.agent.vehicle.transform.rotation.roll) <= .35:
        #         out = 6 * np.exp(-0.05 * np.abs(vehroll))-(la_err/180)*current_speed*kla
        #     else:
        #         out = 2 * np.exp(-0.05 * np.abs(vehroll))-(la_err/180)*current_speed*kla # *****ALGORITHM*****
       # *** calculated formula ***
        else:
            if abs(self.agent.vehicle.transform.rotation.roll) <= 1.2:
                out = 2 * np.exp(-.03 * np.abs(vehroll))-la_err*current_speed*kla
            else:
                out = np.exp(-.06 * np.abs(vehroll))-la_err*current_speed*kla # *****ALGORITHM*****

        output = np.clip(out, a_min=0, a_max=1)
        print('*************')
        print('vehroll:',vehroll)
        print('unclipped throttle = ',out)
        print('throttle = ', output)
        print('*************')


        return output

    def la_calcs(self, next_waypoint: Transform, **kwargs):

        current_speed = int(Vehicle.get_speed(self.agent.vehicle))
        cs = np.clip(current_speed, 70, 200)
        # *** next points on path
        # *** averaging path points for smooth path vector ***

        #la_indx = 8
        la_indx = 43 #coarse points
        #la_indx = 1 # old ROAR map %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # next_pathpoint1 = (self.agent.local_planner.way_points_queue[2*cs+1])
        # next_pathpoint2 = (self.agent.local_planner.way_points_queue[2*cs+2])
        # next_pathpoint3 = (self.agent.local_planner.way_points_queue[2*cs+3])
        # next_pathpoint4 = (self.agent.local_planner.way_points_queue[2*cs+91])
        # next_pathpoint5 = (self.agent.local_planner.way_points_queue[2*cs+92])
        # next_pathpoint6 = (self.agent.local_planner.way_points_queue[2*cs+93])

        # next_pathpoint1 = (self.agent.local_planner.way_points_queue[math.ceil((2*cs+1)/la_indx)])
        # next_pathpoint2 = (self.agent.local_planner.way_points_queue[math.ceil((2*cs+2)/la_indx)])
        # next_pathpoint3 = (self.agent.local_planner.way_points_queue[math.ceil((2*cs+3)/la_indx)])
        # next_pathpoint4 = (self.agent.local_planner.way_points_queue[math.ceil((3*cs+51)/la_indx)])
        # next_pathpoint5 = (self.agent.local_planner.way_points_queue[math.ceil((3*cs+52)/la_indx)])
        # next_pathpoint6 = (self.agent.local_planner.way_points_queue[math.ceil((3*cs+53)/la_indx)])

        lf1 = math.ceil(2*cs/la_indx)
        lf2 = math.ceil(3*cs/la_indx)
        # print ('^^^^^^^^^^^^^^^^^^lf2^^^^^^^^^^^^^^^^^^',lf2)
        print ('+++++++++++ curr wp indx: ',self.agent.local_planner.get_curr_waypoint_index()+lf2+4)
        print ('length wp queue',len(self.agent.local_planner.way_points_queue) )
        if self.agent.local_planner.get_curr_waypoint_index()+lf2+4<=\
            len(self.agent.local_planner.way_points_queue):

            next_pathpoint1 = (self.agent.local_planner.way_points_queue\
                [self.agent.local_planner.get_curr_waypoint_index()+lf1])
            next_pathpoint2 = (self.agent.local_planner.way_points_queue\
                [self.agent.local_planner.get_curr_waypoint_index()+lf1+1])
            next_pathpoint3 = (self.agent.local_planner.way_points_queue\
                [self.agent.local_planner.get_curr_waypoint_index()+lf1+2])
            next_pathpoint4 = (self.agent.local_planner.way_points_queue\
                [self.agent.local_planner.get_curr_waypoint_index()+lf2+1])
            next_pathpoint5 = (self.agent.local_planner.way_points_queue\
                [self.agent.local_planner.get_curr_waypoint_index()+lf2+2])
            next_pathpoint6 = (self.agent.local_planner.way_points_queue\
                [self.agent.local_planner.get_curr_waypoint_index()+lf2+3])


            print('next waypoint: ', self.agent.local_planner.way_points_queue[self.agent.local_planner.get_curr_waypoint_index()])
            print('$$$$$$$$$$$$$way points length: ',self.agent.local_planner.get_curr_waypoint_index(),'/',len(self.agent.local_planner.way_points_queue))

            # ******************************
            # next_pathpoint4 = (self.agent.local_planner.way_points_queue[cs+43])
            # next_pathpoint5 = (self.agent.local_planner.way_points_queue[cs+42])
            # next_pathpoint6 = (self.agent.local_planner.way_points_queue[cs+41])
            # next_pathpoint1 = (self.agent.local_planner.way_points_queue[31])
            # next_pathpoint2 = (self.agent.local_planner.way_points_queue[32])
            # next_pathpoint3 = (self.agent.local_planner.way_points_queue[33])
            # next_pathpoint4 = (self.agent.local_planner.way_points_queue[52])
            # next_pathpoint5 = (self.agent.local_planner.way_points_queue[53])
            # next_pathpoint6 = (self.agent.local_planner.way_points_queue[54])
            nx0 = next_pathpoint1.location.x
            nz0 = next_pathpoint1.location.z
            nx = (
                             next_pathpoint1.location.x + next_pathpoint2.location.x + next_pathpoint3.location.x + next_pathpoint4.location.x + next_pathpoint5.location.x + next_pathpoint6.location.x) / 6
            nz = (
                             next_pathpoint1.location.z + next_pathpoint2.location.z + next_pathpoint3.location.z + next_pathpoint4.location.z + next_pathpoint5.location.z + next_pathpoint6.location.z) / 6
            nx1 = (next_pathpoint1.location.x + next_pathpoint2.location.x + next_pathpoint3.location.x) / 3
            nz1 = (next_pathpoint1.location.z + next_pathpoint2.location.z + next_pathpoint3.location.z) / 3
            nx2 = (next_pathpoint4.location.x + next_pathpoint5.location.x + next_pathpoint6.location.x) / 3
            nz2 = (next_pathpoint4.location.z + next_pathpoint5.location.z + next_pathpoint6.location.z) / 3

            npath0 = np.transpose(np.array([nx0, nz0, 1]))
            npath = np.transpose(np.array([nx, nz, 1]))
            npath1 = np.transpose(np.array([nx1, nz1, 1]))
            npath2 = np.transpose(np.array([nx2, nz2, 1]))

            path_yaw_rad = -(math.atan2((nx2 - nx1), -(nz2 - nz1)))

            path_yaw = path_yaw_rad * 180 / np.pi
            print(' !!! path yaw !!! ', path_yaw)

            veh_yaw = self.agent.vehicle.transform.rotation.yaw
            print(' !!! veh yaw  !!! ', veh_yaw)
            ahead_err = abs(abs(path_yaw)-abs(veh_yaw))

        else:
            ahead_err = 105

        if ahead_err < 60:
            la_err = 0
        else:
            la_err =(.05 * ahead_err)**3

        print('--------------------------------------')


        print('** la err **', la_err)
        print('--------------------------------------')

        return la_err

        #***********************************************************

# ***** end original version Roll ContRoller *****


class LatPIDController(Controller):
    def __init__(self, agent, config: dict, steering_boundary: Tuple[float, float],
                 dt: float = 0.03, **kwargs):
        super().__init__(agent, **kwargs)
        self.config = config
        self.steering_boundary = steering_boundary
        self._error_buffer = deque(maxlen=10)
        self._dt = dt

    def run_in_series(self, next_waypoint: Transform, **kwargs) -> float:
        """
        Calculates a vector that represent where you are going.
        Args:
            next_waypoint ():
            **kwargs ():
        Returns:
            lat_control
        """
        # calculate a vector that represent where you are going
        v_begin = self.agent.vehicle.transform.location.to_array()

        print(v_begin)
        print('next wp x: ', next_waypoint.location.x)
        print('next wp z: ', next_waypoint.location.z)
        print('next wp y: ', next_waypoint.location.y)

        direction_vector = np.array([-np.sin(np.deg2rad(self.agent.vehicle.transform.rotation.yaw)),
                                     0,
                                     -np.cos(np.deg2rad(self.agent.vehicle.transform.rotation.yaw))])

        v_end = v_begin + direction_vector

        v_vec = np.array([(v_end[0] - v_begin[0]), 0, (v_end[2] - v_begin[2])])
        # calculate error projection
        w_vec = np.array(
            [
                next_waypoint.location.x - v_begin[0],
                0,
                next_waypoint.location.z - v_begin[2],
            ]
        )

        v_vec_normed = v_vec / np.linalg.norm(v_vec)
        w_vec_normed = w_vec / np.linalg.norm(w_vec)
        error = np.arccos(v_vec_normed @ w_vec_normed.T)
        _cross = np.cross(v_vec_normed, w_vec_normed)

        if _cross[1] > 0:
            error *= -1
        self._error_buffer.append(error)
        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        k_p, k_d, k_i = PIDRollController.find_k_values(config=self.config, vehicle=self.agent.vehicle)
        print ('kp, kd, ki: ', k_p, k_d, k_i)
        lat_control = float(
             np.clip((k_p * error) + (k_d * _de) + (k_i * _ie), self.steering_boundary[0], self.steering_boundary[1])
        )
        # lat_control = float(
        #     np.clip((k_p * error) + (k_d * _de) + (k_i * _ie), -.9, .9)
        # )
        # print(f"v_vec_normed: {v_vec_normed} | w_vec_normed = {w_vec_normed}")
        # print("v_vec_normed @ w_vec_normed.T:", v_vec_normed @ w_vec_normed.T)
        # print(f"Curr: {self.agent.vehicle.transform.location}, waypoint: {next_waypoint}")
        # print(f"lat_control: {round(lat_control, 3)} | error: {error} ")
        # print()
        # print('steering boundary:', self.steering_boundary)
        print('lateral control:',lat_control)
        return lat_control