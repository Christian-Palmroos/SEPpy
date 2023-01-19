
import numpy as np
import pandas as pd
import astropy.constants as const
import astropy.units as u
import sunpy.sun.constants as sconst

import datetime
from sunpy.coordinates import get_horizons_coord

# Utilities toolbox, contains helpful functions


def resample_df(df, resample, pos_timestamp="center", origin="start"):
    """
    Resamples a Pandas Dataframe or Series to a new frequency.

    Parameters:
    -----------
    df : pd.DataFrame or pd.Series
            The dataframe or series to resample
    resample : str
            pandas-compatible time string, e.g., '1min', '2H' or '25s'
    pos_timestamp : str, default 'center'
            Controls if the timestamp is at the center of the time bin, or at the start of it
    origin : str, default 'start'
            Controls if the origin of resampling is at the start of the day (midnight) or at the first
            entry of the input dataframe/series

    Returns:
    ----------
    df : pd.DataFrame or Series, depending on the input
    """
    try:
        df = df.resample(resample, origin=origin, label="left").mean()
        if pos_timestamp == 'start':
            df.index = df.index
        else:
            df.index = df.index + pd.tseries.frequencies.to_offset(pd.Timedelta(resample)/2)
        # if pos_timestamp == 'stop' or pos_timestamp == 'end':
        #     df.index = df.index + pd.tseries.frequencies.to_offset(pd.Timedelta(resample))
    except ValueError:
        raise ValueError(f"Your 'resample' option of [{resample}] doesn't seem to be a proper Pandas frequency!")

    return df


def flux2series(flux, dates, cadence=None):
    """
    Converts an array of observed particle flux + timestamps into a pandas series
    with the desired cadence.

    Parameters:
    -----------
    flux: an array of observed particle fluxes
    dates: an array of corresponding dates/times
    cadence: str - desired spacing between the series elements e.g. '1s' or '5min'

    Returns:
    ----------
    flux_series: Pandas Series object indexed by the resampled cadence
    """

    # from pandas.tseries.frequencies import to_offset

    # set up the series object
    flux_series = pd.Series(flux, index=dates)

    # if no cadence given, then just return the series with the original
    # time resolution
    if cadence is not None:
        flux_series = resample_df(df=flux_series, resample=cadence, pos_timestamp="center", origin="start")

    return flux_series


def bepicolombo_sixs_stack(path, date, side):
    # side is the index of the file here
    try:
        try:
            filename = f"{path}/sixs_phys_data_{date}_side{side}.csv"
            df = pd.read_csv(filename)
        except FileNotFoundError:
            # try alternative file name format
            filename = f"{path}/{date.strftime('%Y%m%d')}_side{side}.csv"
            df = pd.read_csv(filename)
            times = pd.to_datetime(df['TimeUTC'])
        # list comprehension because the method can't be applied onto the array "times"
        times = [t.tz_convert(None) for t in times]
        df.index = np.array(times)
        df = df.drop(columns=['TimeUTC'])
    except FileNotFoundError:
        print(f'Unable to open {filename}')
        df = pd.DataFrame()
        filename = ''
    return df, filename


def bepi_sixs_load(startdate, enddate, side, path):
    dates = pd.date_range(startdate, enddate)

    # read files into Pandas dataframes:
    df, file = bepicolombo_sixs_stack(path, startdate, side=side)
    if len(dates) > 1:
        for date in dates[1:]:
            t_df, file = bepicolombo_sixs_stack(path, date.date(), side=side)
            df = pd.concat([df, t_df])

    channels_dict = {"Energy_Bin_str": {'E1': '71 keV', 'E2': '106 keV', 'E3': '169 keV', 'E4': '280 keV', 'E5': '960 keV', 'E6': '2240 keV', 'E7': '8170 keV',
                                        'P1': '1.1 MeV', 'P2': '1.2 MeV', 'P3': '1.5 MeV', 'P4': '2.3 MeV', 'P5': '4.0 MeV', 'P6': '8.0 MeV', 'P7': '15.0 MeV', 'P8': '25.1 MeV', 'P9': '37.3 MeV'},
                     "Electron_Bins_Low_Energy": np.array([55, 78, 134, 235, 1000, 1432, 4904]),
                     "Electron_Bins_High_Energy": np.array([92, 143, 214, 331, 1193, 3165, 10000]),
                     "Ion_Bins_Low_Energy": np.array([0.001, 1.088, 1.407, 2.139, 3.647, 7.533, 13.211, 22.606, 29.246]),
                     "Ion_Bins_High_Energy": np.array([1.254, 1.311, 1.608, 2.388, 4.241, 8.534, 15.515, 28.413, 40.0])}
    return df, channels_dict


def calc_av_en_flux_sixs(df, channel, species):
    """
    This function averages the flux of two energy channels of BepiColombo/SIXS into a combined energy channel
    channel numbers counted from 1

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing HET data
    channel : int or list
        energy channel or list with first and last channel to be used
    species : string
        'e', 'electrons', 'p', 'protons'

    Returns
    -------
    flux: pd.DataFrame
        channel-averaged flux
    en_channel_string: str
        string containing the energy information of combined channel
    """

    # define constant geometric factors
    GEOMFACTOR_PROT8 = 5.97E-01
    GEOMFACTOR_PROT9 = 4.09E+00
    GEOMFACTOR_ELEC5 = 1.99E-02
    GEOMFACTOR_ELEC6 = 1.33E-01
    GEOMFACTOR_PROT_COMB89 = 3.34
    GEOMFACTOR_ELEC_COMB56 = 0.0972

    if species in ['p', 'protons']:
        if channel == [8, 9]:
            countrate = df['P8'] * GEOMFACTOR_PROT8 + df['P9'] * GEOMFACTOR_PROT9
            flux = countrate / GEOMFACTOR_PROT_COMB89
            en_channel_string = '37 MeV'
        else:
            print('No valid channel combination selected.')
            flux = pd.Series()
            en_channel_string = ''

    if species in ['e', 'electrons']:
        if channel == [5, 6]:
            countrate = df['E5'] * GEOMFACTOR_ELEC5 + df['E6'] * GEOMFACTOR_ELEC6
            flux = countrate / GEOMFACTOR_ELEC_COMB56
            en_channel_string = '1.4 MeV'
        else:
            print('No valid channel combination selected.')
            flux = pd.Series()
            en_channel_string = ''

    return flux, en_channel_string


"""
inf_inj_time.py
"""
SOLAR_ROT = sconst.get('sidereal rotation rate').to(u.rad/u.s)


def get_sun_coords(time='now'):
    '''
    Gets the astropy Sun coordinates.

    Args:
        time (datetime.datetime): time at which coordinates are fetched.

    Returns:
        sun coordinates.
    '''

    return get_horizons_coord("Sun", time=time)


def radial_distance_to_sun(spacecraft, time='now'):
    '''
    Gets the 3D radial distance of a spacecraft to the Sun.
    3D here means that it's the real spatial distance and not
    a projection on, say, the solar equatorial plane.

    Args:
        spacecraft (str): spacecraft to look for.
        time (datetime.datetime): time at which to look for.

    Returns:
        astropy units: radial distance.
    '''

    sc_coords = get_horizons_coord(spacecraft, time)

    return sc_coords.separation_3d(get_sun_coords(time=time))


def calc_spiral_length(radial_dist, sw_speed):
    '''
    Calculates the Parker spiral length from the Sun up to a given radial distance.

    Args:
        radial_dist (astropy units): radial distance to the Sun.
        sw_speed (astropy units): solar wind speed.

    Returns:
        astropy units: Parker spiral length.
    '''

    temp_const = ((SOLAR_ROT/sw_speed)*(radial_dist.to(u.km)-const.R_sun)).value
    sqrt_temp_const = np.sqrt(temp_const**2 + 1)

    return 0.5*u.rad * (sw_speed/SOLAR_ROT) * (temp_const*sqrt_temp_const + np.log(temp_const + sqrt_temp_const))


def calc_particle_speed(mass, kinetic_energy):
    '''
    Calculates the relativistic particle speed.

    Args:
        mass (astropy units): mass of the particle.
        kinetic_energy (astropy units): kinetic energy of the particle.

    Returns:
        astropy units: relativistic particle speed.
    '''

    gamma = np.sqrt(1 - (mass*const.c**2/(kinetic_energy + mass*const.c**2))**2)

    return gamma*const.c


def inf_inj_time(spacecraft, onset_time, species, kinetic_energy, sw_speed):
    '''
    Calculates the inferred injection time of a particle (electron or proton) from the Sun,
    given a detection time at some spacecraft.

    Args:
        spacecraft (str): name of the spacecraft.
        onset_time (datetime.datetime): time of onset/detection.
        species (str): particle species, 'p' or 'e'.
        kinetic_energy (astropy units): kinetic energy of particle. If no unit is supplied, is converted to MeV.
        sw_speed (astropy units): solar wind speed. If no unit is supplied, is converted to km/s.

    Returns:
        datetime.datetime: inferred injection time.
    '''

    if not type(kinetic_energy)==u.quantity.Quantity:
        kinetic_energy = kinetic_energy * u.MeV

    if not type(sw_speed)==u.quantity.Quantity:
        sw_speed = sw_speed * u.km/u.s

    mass_dict = {'p': const.m_p,
                 'e': const.m_e
                 }

    radial_distance = radial_distance_to_sun(spacecraft, time=onset_time)

    spiral_length = calc_spiral_length(radial_distance, sw_speed)
    particle_speed = calc_particle_speed(mass_dict[species], kinetic_energy)

    travel_time = spiral_length/particle_speed
    travel_time = travel_time.to(u.s)

    return onset_time - datetime.timedelta(seconds=travel_time.value), spiral_length.to(u.AU)
