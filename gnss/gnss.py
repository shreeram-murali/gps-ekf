import numpy as np
from gps_helper import gps_helper as GPS
from datetime import datetime, timezone

def random_normal(mu, sigma, samples=None, seed=None):
    """
    A wrapper to sample from a normal distribution. Uses the new Generator class from NumPy. 
    The seed value is randomly sampled from the OS if it is set as None. 
    Docuhttps://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.normal.html
    ---
    Input:
        mu: mean of the normal distribution
        sigma: standard deviation 
        samples: if None, returns 1 number
    Output:
        random sample or a an array of samples
    """
    
    return np.random.default_rng(seed).normal(mu, sigma, samples)
    

class SimulatePseudorange:
    """
    A class for generating simulated pseudorange measurements. 
    This module generates pseudoranges based on simulated satellite vehicle movement as well as GPS receiver motion.
    """

    def __init__(self, pr_std = 0, cdt = 0, nsats = 4, dt=1):
        """
            pr_std (float): pseudorange measureemnt noise standard deviation 
            cdt (float): clock bias 
            nsats (int): number of satellites
        """

        self.pr_std = pr_std
        self.cdt = cdt
        self.nsats = nsats
        self.user_pseudorange = np.zeros((nsats, 1))

        self.GPS_dataset = self.create_gps_dataset(dt)

    def measurement(self, user_pos, sat_pos) -> None:
        """
        Measurements based on position at a given time interval user_pos[i,:] and sat_pos[:,:,i]
        ---
        Inputs:
            user_pos (np.array): user position at time i
            sat_pos (np.array): satellite position at time i
        Outputs:
            Returns nothing. Effectful function that updates self.user_pseudorange.
        """

        for i in range(self.nsats):
            self.user_pseudorange[i, 0] = np.linalg.norm(user_pos - sat_pos[i])
            self.user_pseudorange[i, 0] += self.cdt + random_normal(0, self.pr_std)
    
    def generate_usersatpos(self, line_segment, user_velocity, today=datetime.now(timezone.utc)):
        """
        Generates user positions in ECEF and ENU coordinate frames, satellite positions, and satellite velocities
        ---
        Inputs:
            line_segment (iterable): line segment in ENU coordinate frame
            user_velocity (float): user velocity (meters per hour)
            today (datetime object): current time
            Ts (int): time step
        Outputs:
            user_pos_ecef, user_pos_enu, sat_pos, sat_vel
        """

        userpos_enu, userpos_ecf, satpos, satvel = self.GPS_dataset.user_traj_gen(
            route_list=line_segment, 
            vmph=user_velocity, 
            yr2=today.year - 2000,
            mon=today.month, 
            day=today.day, 
            hr=today.hour, 
            minute=today.minute,
            sec=today.second
        )

        return userpos_enu, userpos_ecf, satpos, satvel
    
    def traj3d_viz(self, satpos, userpos_ecf, ele=20, azi=-40):
        return GPS.sv_user_traj_3d(self.GPS_dataset, satpos, userpos_ecf, ele, azi)
    
    def ecef2enu(self, pos):
        refecef, lla1, lla2 = self.GPS_dataset.ref_ecef, self.GPS_dataset.ref_lla[0], self.GPS_dataset.ref_lla[1]
        return GPS.ecef2enu(pos, refecef, lla1, lla2)
    
    def create_gps_dataset(self, Ts):
        ds = GPS.GPSDataSource('GPS_tle_1_10_2018.txt', rx_sv_list = ('PRN 32','PRN 21','PRN 10','PRN 18'),
                                ref_lla=(38.8454167, -104.7215556, 1903.0), ts=Ts)
        return ds
    