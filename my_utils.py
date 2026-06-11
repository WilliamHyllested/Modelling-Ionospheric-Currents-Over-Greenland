import os
import cdflib # for cdf load 
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time, date
import matplotlib.pyplot as plt
import glob
from typing import Dict, Union
from chaosmagpy import load_CHAOS_matfile
from chaosmagpy.coordinate_utils import transform_points
from chaosmagpy.coordinate_utils import geo_to_gg, gg_to_geo
from chaosmagpy.data_utils import mjd2000, load_RC_datfile

## Convert coordinates and components from geodetic -> geocentric 
# By marvig, DTU Space
def gd2gc(lat_gd, h, X_gd, Z_gd):
    
    # Input:
    #      lat_gd = geodetic latitude in degrees
    #      h =      altitude in km
    #      X_gd =   geodetic north component
    #      Z_gd =   geodetic vertical component
    
    # Output:
    #      r =        radius in km
    #      colat_gc = geocentric co-latitude in degrees
    #      Z_gc =     geocentric vertical component
    #      X_gc =     geocentric north component
    
    # Ellipsoid GRS 80 (identical for WGS84)
    a = 6378.137 
    b = 6356.752
    
    sin_alpha_2 = np.sin(np.deg2rad(lat_gd))**2
    cos_alpha_2 = np.cos(np.deg2rad(lat_gd))**2
    
    tmp = h*np.sqrt(a**2*cos_alpha_2 + b**2*sin_alpha_2)
    beta = np.arctan((tmp+b**2)/(tmp+a**2)*np.tan(np.deg2rad(lat_gd)))
    
    # Geocentric co-latitude in degrees
    colat_gc = (np.pi/2 - beta)*180/np.pi
    
    # Radius in km
    r = np.sqrt(h**2 + 2*tmp + a**2*(1 - (1 - (b/a)**4)*sin_alpha_2)/(1 - (1 - (b/a)**2)*sin_alpha_2))
    
    # convert also magnetic components
    psi = np.sin(np.deg2rad(lat_gd))*np.sin(np.deg2rad(colat_gc)) - np.cos(np.deg2rad(lat_gd))*np.cos(np.deg2rad(colat_gc))
    
    # Geocentric radial component
    B_r_gc = -np.sin(psi)*X_gd - np.cos(psi)*Z_gd
    
    # Geocentric theta component
    B_theta_gc = -np.cos(psi)*X_gd + np.sin(psi)*Z_gd
    
    # Convert to X and Z
    Z_gc = -B_r_gc
    X_gc = -B_theta_gc
    
    return r, colat_gc, Z_gc, X_gc



# Created by marvig, 
# 16/12/2022
def gc2gd(radius_gc, theta_gc, X_gc, Z_gc):
    
    # INPUT:
    #      radius_gc: Geocentric radius in km
    #      theta_gc: Geocentric colatitude in degrees
    #      X_gc: Geocentric X-component
    #      Z_gc: Geocentric Z-component
    #
    # OUTPUT:
    #      height: Altitude in km (height above ellipsoid)
    #      beta: Geodetic colatitude in degrees
    #      X_gd: Geodetic X-component
    #      Z_gd: Geodetic Z-component

    # Get height above the ellipsoid in km and geodetic co-latitude in degrees from CHAOS
    height, beta = geo_to_gg(radius_gc, theta_gc)

    # Convert the magnetic field components as well
    psi = np.sin(np.deg2rad(90-beta))*np.sin(np.deg2rad(theta_gc))-np.cos(np.deg2rad(90-beta))*np.cos(np.deg2rad(theta_gc))

    X_gd = -np.cos(psi)*(-X_gc)-np.sin(psi)*(-Z_gc)
    Z_gd = np.sin(psi)*(-X_gc)-np.cos(psi)*(-Z_gc)
    
    return height, beta, X_gd, Z_gd


"""
Convert datetime objects to Modified Julian Date 2000 (MJD2000).

MJD2000 is the number of days since 2000-01-01 00:00:00 UTC.

Supports:
  - Single datetime objects
  - Lists of datetime objects
  - numpy arrays of datetime64
"""

from datetime import datetime, timedelta, timezone
from typing import Union
import numpy as np


EPOCH = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
EPOCH_NP = np.datetime64("2000-01-01T00:00:00", "ns")


def datetime_to_mjd2000(dt: Union[datetime, list, np.ndarray]) -> Union[float, list, np.ndarray]:
    """
    Convert datetime(s) to MJD2000.

    MJD2000 is defined as the number of days elapsed since
    2000-01-01 00:00:00.000 UTC.

    Args:
        dt: One of:
            - A single datetime object
            - A list of datetime objects
            - A numpy array of datetime64

    Returns:
        - float for a single datetime
        - list of floats for a list input
        - numpy array of float64 for a numpy array input
    """
    # numpy array of datetime64
    if isinstance(dt, np.ndarray):
        arr = dt.astype("datetime64[ns]")
        return (arr - EPOCH_NP).astype("float64") / 86_400e9

    # list of datetimes
    if isinstance(dt, list):
        return [datetime_to_mjd2000(d) for d in dt]

    # single datetime
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt - EPOCH).total_seconds() / 86400.0


def mjd2000_to_datetime(mjd2000: Union[float, list, np.ndarray]) -> Union[datetime, list, np.ndarray]:
    """
    Convert MJD2000 value(s) back to datetime(s), rounded to the nearest second.

    Args:
        mjd2000: One of:
            - A single float
            - A list of floats
            - A numpy array of floats

    Returns:
        - A UTC datetime for a single float
        - A list of UTC datetimes for a list input
        - A numpy array of datetime64 for a numpy array input
    """
    if isinstance(mjd2000, np.ndarray):
        ns = (mjd2000 * 86_400e9).astype("int64")
        dt64 = EPOCH_NP + ns.astype("timedelta64[ns]")
        return dt64.astype("datetime64[s]")

    if isinstance(mjd2000, list):
        return [mjd2000_to_datetime(v) for v in mjd2000]

    seconds = round(mjd2000 * 86_400)
    return EPOCH + timedelta(seconds=seconds)



## For some reason cdflib won't work without this function (for a specific version of python)
def vdr_info(self, variable: Union[str, int]):
        if (isinstance(variable, int) and self._num_zvariable > 0 and
                self._num_rvariable > 0):
            raise ValueError('This CDF has both r and z variables. '
                             'Use variable name instead')

        if isinstance(variable, str):
            # Check z variables for the name, then r variables
            position = self._first_zvariable
            num_variables = self._num_zvariable
            vdr_info = None
            for zVar in [1, 0]:
                for _ in range(0, num_variables):
                    name, vdr_next = self._read_vdr_fast(position)
                    if name.strip().lower() == variable.strip().lower():
                        vdr_info = self._read_vdr(position)
                        break
                    position = vdr_next
                position = self._first_rvariable
                num_variables = self._num_rvariable
            if vdr_info is None:
                raise ValueError(f"Variable name '{variable}' not found.")
        elif isinstance(variable, int):
            if self._num_zvariable > 0:
                position = self._first_zvariable
                num_variable = self._num_zvariable
                # zVar = True
            elif self._num_rvariable > 0:
                position = self._first_rvariable
                num_variable = self._num_rvariable
                # zVar = False
            if (variable < 0 or variable >= num_variable):
                raise ValueError(
                    f'No variable by this number: {variable}')
            for _ in range(0, variable):
                name, next_vdr = self._read_vdr_fast(position)
                position = next_vdr
            vdr_info = self._read_vdr(position)
        else:
            raise ValueError('Please set variable keyword equal to '
                             'the name or number of an variable')

        return vdr_info


def load_cdf_file(filepath):

    cdf_file = cdflib.CDF(filepath)

    #if vdr_info(cdf_file, 'HNvar')['head_vxr'] > 0 and (len(cdf_file['HNvar'])==86400):

    # Component sensor X
    X_s = cdf_file['HNscv'][0]*cdf_file['HNvar']+cdf_file['H0'][0]

    # Component sensor Y
    Y_s = cdf_file['HEscv'][0]*cdf_file['HEvar']

    # Radial sensor component in nT
    Z_s = cdf_file['Zvar']*cdf_file['Zscv'][0]+cdf_file['Z0'][0]

    # Horizontal component in nT
    H = np.sqrt(X_s**2+Y_s**2)

    # Declination in deg
    D = np.arctan2(Y_s,X_s) + cdf_file['D0'][0]*np.pi/180 # rad

    # Component pointing towards geographic north
    X = H*np.cos(D)

    # Component point towards geographic east
    Y = H*np.sin(D)

    # Radial component Z
    Z = cdf_file['Zvar']*cdf_file['Zscv'][0]+cdf_file['Z0'][0]

    # Time in MJD2000
    t_mjd = cdf_file['time']

    ## Quality control
    try:
        X[cdf_file['HNflag'] != 0] = np.nan
        Y[cdf_file['HEflag'] != 0] = np.nan
        Z[cdf_file['Zflag'] != 0] = np.nan
        
    except AttributeError:
        print('No quality control found')


    return X, Y, Z, t_mjd

def load_folder_to_dataframe(folder_path, minute = False):
    all_dfs = []

    files = sorted([
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ])

    print(f"Found {len(files)} files in '{folder_path}'")

    for filename in files:
        filepath = os.path.join(folder_path, filename)
        #print(f"Loading: {filename}")

        try:
            X, Y, Z, t_mjd = load_cdf_file(filepath)
            
            df = pd.DataFrame({
                'time_mjd': t_mjd,
                'X': X,
                'Y': Y,
                'Z': Z,
                'source_file': filename
            })

            if minute==True:
                df = df[::60]

            all_dfs.append(df)

        except Exception as e:
            print(f"  Warning: Failed to load {filename}: {e}")

    if not all_dfs:
        print("No data was loaded.")
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nDone! Combined DataFrame shape: {combined_df.shape}")
    return combined_df


def load_and_combine_dtu(date_str, folders, base_dir='dtu_downloads/yearly_BASELINE_rotated'):
    """
    Load files matching a date string across multiple folders and combine into one DataFrame.
    
    Args:
        date_str:     Date string like '20250214'
        folder_names: List of folder name strings
        base_dir:     Base directory containing all folders
    
    Returns:
        Combined DataFrame with data from all folders
    """
    combined_df = []

    year = date_str[0:4]
    month = date_str[4:6]
    if int(month) < 10:
            month = month[-1]
    day = date_str[6:8]

    for folder in folders:
        sta = folder[0:3]
        filename = f"dtu_{sta}_{year}_{month}.parquet"

        filepath = os.path.join(base_dir, folder, filename)

        if not os.path.exists(filepath):
            print(f"[WARNING] File not found, skipping: {filepath}")
            continue

        df = pd.read_parquet(filepath, engine='fastparquet')
        df['iaga'] = sta
        
        combined_df.append(df)
        
    if not combined_df:
        raise FileNotFoundError(f"No files found for date '{date_str}'")

    combined_df = pd.concat(combined_df, ignore_index=True)

    combined_df['time'] = pd.to_datetime(combined_df['time'])
    combined_df = combined_df.sort_values(
        by=["time", 'iaga'],
        ascending=[True, True]
    ).reset_index(drop=True)

    combined_df = combined_df.drop_duplicates(['time','iaga'])
    
    return combined_df

def load_and_combine_supermag(date_str, stations, base_dir='supermag_downloads/yearly_BASELINE'):
    """
    Load csv files matching a date string across multiple folders and combine into one DataFrame.
    
    Args:
        date_str:     Date string like '20250214'
        folder_names: List of folder name strings
        base_dir:     Base directory containing all folders
    
    Returns:
        Combined DataFrame with data from all folders
    """
    combined_df = []

    year = date_str[0:4]
    month = date_str[4:6]
    if int(month) < 10:
            month = month[-1]
    day = date_str[6:8]

    for sta in stations:
        filename = f"supermag_{sta}_{year}_{month}.csv"

        filepath = os.path.join(base_dir, year, filename)

        if not os.path.exists(filepath):
            print(f"[WARNING] File not found, skipping: {filepath}")
            continue

        df = pd.read_csv(filepath)
        
        combined_df.append(df)
        
    if not combined_df:
        raise FileNotFoundError(f"No files found for date '{date_str}'")

    combined_df = pd.concat(combined_df, ignore_index=True)

    combined_df['time'] = pd.to_datetime(combined_df['time'])
    combined_df = combined_df.sort_values(
        by=["time", 'iaga'],
        ascending=[True, True]
    ).reset_index(drop=True)

    return combined_df


def my_chaos(model, rc, colat_gc, lon_gc, r, t_mjd, df_omni=None, internal=False):

    # give inputs
    theta = np.array([colat_gc]).flatten()  # colat in deg
    phi = np.array([lon_gc]).flatten()  # longitude in deg
    radius = np.array([r]).flatten() # radius from altitude in km
    time = np.array([t_mjd]).flatten()  # time in mjd2000

    # Interpolate RC-index onto input time
    rc_e = np.interp(time, rc['time'], rc['RC_e'])
    rc_i = np.interp(time, rc['time'], rc['RC_i'])

    print('Computing core field.')
    B_core = model.synth_values_tdep(time, radius, theta, phi)

    print('Computing crustal field up to degree 110.')
    B_crust = model.synth_values_static(radius, theta, phi, nmax=110)

    # complete internal contribution
    B_radius_int = B_core[0] + B_crust[0]
    B_theta_int = B_core[1] + B_crust[1]
    B_phi_int = B_core[2] + B_crust[2]

    print('Computing field due to external sources, incl. induced field: GSM.')
    B_gsm = model.synth_values_gsm(time, radius, theta, phi, source='all')

    print('Computing field due to external sources, incl. induced field: SM.')
    B_sm = model.synth_values_sm(time, radius, theta, phi,
                                    rc_e=rc_e, rc_i=rc_i, source='all')
    
    if df_omni is not None:
        print('Computing field due to external sources, incl. ionospheric field.')
        f107=100*np.ones(time.size)
        B_ion = model.synth_values_ion(time, radius, theta, phi, imf_y=df_omni['by'].values, imf_z=df_omni['bz'].values, v=df_omni['plasma_speed'].values,
                            f107=f107)
        # complete external field contribution
        B_radius_ext = B_gsm[0] + B_sm[0] + B_ion[0]
        B_theta_ext = B_gsm[1] + B_sm[1] + B_ion[1]
        B_phi_ext = B_gsm[2] + B_sm[2] + B_ion[2]
    else:
        # complete external field contribution
        B_radius_ext = B_gsm[0] + B_sm[0]
        B_theta_ext = B_gsm[1] + B_sm[1]
        B_phi_ext = B_gsm[2] + B_sm[2]

    # complete forward computation
    B_radius = B_radius_int + B_radius_ext
    B_theta = B_theta_int + B_theta_ext
    B_phi = B_phi_int + B_phi_ext
    if internal==True:
        B_radius = B_radius_int
        B_theta = B_theta_int
        B_phi = B_phi_int
        
    # Convert CHAOS to geodetic components
    _, _, Bx_gd, Bz_gd = gc2gd(radius, theta, -B_theta, -B_radius)
    By_gd = B_phi

    # save to output file
    df_CHAOS = pd.DataFrame(
        np.stack([time, radius, theta, phi, B_radius, B_theta, B_phi, Bx_gd, By_gd, Bz_gd], axis=-1),
        columns=['time', 'radius', 'theta', 'phi', 'B_radius', 'B_theta', 'B_phi', 'Bx_gd', 'By_gd', 'Bz_gd']
    #df_CHAOS = pd.DataFrame(
    #    np.stack([time, radius, theta, phi, B_radius, B_theta, B_phi,
    #          B_radius_int, B_theta_int, B_phi_int, B_radius_ext,
    #          B_theta_ext, B_phi_ext], axis=-1),
    #    columns=['time', 'radius', 'theta', 'phi', 'B_radius', 'B_theta', 'B_phi',
    #         'B_radius_int', 'B_theta_int', 'B_phi_int', 'B_radius_ext',
    #         'B_theta_ext', 'B_phi_ext']
    )
    
    return df_CHAOS


import re
from io import StringIO
import dipole

def load_omni(file_path, times=None, hourly=False, second = False):

    filename_omni = file_path

    # 1. Read and clean the raw text of the file
    with open(filename_omni, 'r') as f:
        raw_text = f.read()

    # Remove the markers that cause column alignment issues
    clean_text = re.sub(r'\\', '', raw_text)

    # 2. Load into Pandas, treating placeholders as NaN
    df = pd.read_csv(
        StringIO(clean_text),
        sep=r'\s+',
        header=None,
        na_values=['9999.99', '99999.9']
    )

    # Assign standard OMNI column names
    df.columns = [
        'year', 'day', 'hour', 'minute', 
        'IMF_magnitude', 'by', 'bz', 
        'IMF_magnitude_sd', 'plasma_speed'
    ]

    # Combine year, day of year, hour, and minute into a single timestamp
    df['timestamp'] = pd.to_datetime(df['year'], format='%Y') + \
                    pd.to_timedelta(df['day'] - 1, unit='D') + \
                    pd.to_timedelta(df['hour'], unit='h') + \
                    pd.to_timedelta(df['minute'], unit='m')

    df.set_index('timestamp', inplace=True)
    df = df.sort_index()

    # 3. Fill the NaN values
    # 'linear' interpolation is best for magnetic fields and plasma speed
    # 'limit_direction="both"' ensures that NaNs at the very start/end are also filled
    df_omni = df.interpolate(method='linear', limit_direction='both')

    # smooth with 20min window (Kalle told me to do this)
    cols_to_smooth = ['IMF_magnitude', 'by', 'bz', 'IMF_magnitude_sd', 'plasma_speed']
    # Apply smoothing only to these specific columns
    df_omni[cols_to_smooth] = df_omni[cols_to_smooth].rolling('20Min').mean().values

    # add IMF clock angle (By and Bz must be in GSM)
    df_omni['clock_angle'] = np.arctan2(df_omni['by'], df_omni['bz'])*180/np.pi

    # add dipole tilt angle
    Epoch = df_omni.index[0].year
    df_omni['tilts'] = dipole.Dipole(Epoch).tilt(df_omni.index)

    if second == True:
        df_omni = df_omni.resample('1s').ffill()

    # remove times not present in df_mag
    if times is not None:
        df_omni = df_omni[df_omni.index.isin(times)]

    #downsample to hourly data if
    if hourly == True:
        df_omni = df_omni.resample('h').mean()

    # mask to remove disturbed data
    mask_omni = (df_omni['by'] > -10) & (df_omni['by'] < 10) & (df_omni['bz'] > -10) & (df_omni['bz'] < 0) & (df_omni['tilts'] < 0)

    return df_omni, mask_omni



###### DOWNSAMPLING INTERMAGNET METHOD AS IMPLEMENTED BY CLAUDE #######

from typing import Literal

# ── Filter coefficients ───────────────────────────────────────────────────────

COEFFS_1S = [
    0.0251958,  0.02514602, 0.02499727, 0.02475132, 0.02441104,
    0.0239804,  0.02346437, 0.02286881, 0.02220039, 0.02146643,
    0.0206748,  0.01983377, 0.01895183, 0.01803763, 0.01709976,
    0.01614667, 0.01518651, 0.01422707, 0.01327563, 0.01233892,
    0.01142303, 0.01053338, 0.0096747,  0.0088509,  0.0080653,
    0.0073204,  0.0066181,  0.0059596,  0.0053454,  0.0047755,
    0.0042496,  0.0037667,  0.0033254,  0.0029243,  0.0025614,
    0.0022347,  0.0019419,  0.0016809,  0.0014492,  0.0012445,
    0.0010645,  0.000907,   0.00077,    0.000651,   0.000548,
    0.000459,
]

COEFFS_5S = [
    0.12578865, 0.11972085, 0.10321785, 0.0806114,  0.05702885,
    0.0365468,  0.02121585, 0.01115655, 0.0053144,  0.0022932,
]

COEFFS_10S = [
    0.25100743, 0.20596804, 0.11379931, 0.04233562, 0.01060471,
]

COEFFS = {"1s": COEFFS_1S, "5s": COEFFS_5S, "10s": COEFFS_10S}


# ── Core filter ───────────────────────────────────────────────────────────────

def _build_kernel(coeffs: list[float]) -> np.ndarray:
    """Build and normalise the full symmetric FIR kernel."""
    kernel = np.array(coeffs[::-1] + coeffs)
    return kernel / kernel.sum()


def _apply_filter_1d(values: np.ndarray, kernel: np.ndarray, nan_threshold: float) -> np.ndarray:
    """
    Apply a symmetric FIR filter to a 1-D array, handling NaNs by:
      1. Replacing NaNs with 0 for the signal convolution.
      2. Convolving a binary valid-sample mask with the same kernel
         to get the sum of weights that actually had data.
      3. Dividing signal by weight sum → correct weighted mean over
         only the non-NaN neighbours.
      4. Masking output positions where valid weight coverage is below
         `nan_threshold` (0–1) back to NaN.
    """
    nan_mask = np.isnan(values)
    signal   = np.where(nan_mask, 0.0, values)
    valid    = (~nan_mask).astype(float)

    conv_signal = np.convolve(signal, kernel, mode="same")
    conv_weight = np.convolve(valid,  kernel, mode="same")

    # Avoid division by zero; positions with no coverage stay NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(conv_weight > 0, conv_signal / conv_weight, np.nan)

    # Mask positions where too little of the kernel window had real data
    result[conv_weight < nan_threshold] = np.nan

    return result


# ── Public API ────────────────────────────────────────────────────────────────

def downsample_to_1min(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    input_rate: Literal["1s", "5s", "10s"] = "1s",
    nan_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Low-pass filter then resample a time-indexed DataFrame to 1-minute values.

    Parameters
    ----------
    df            : DataFrame with a DatetimeIndex at a uniform sample rate.
    columns       : Columns to filter. Defaults to all numeric columns.
    input_rate    : Sample rate of the input data – "1s", "5s", or "10s".
    nan_threshold : Minimum fraction of kernel weight that must come from
                    non-NaN samples for a filtered value to be kept (0–1).
                    E.g. 0.5 means >50 % of the kernel window must be valid.
    """
    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()

    kernel = _build_kernel(COEFFS[input_rate])
    result = df.copy()

    for col in columns:
        result[col] = _apply_filter_1d(
            df[col].to_numpy(dtype=float), kernel, nan_threshold
        )

    return result.resample("1min").first()



def compute_declination(lat_gg, lon, alt_km, time_mjd2000, model):
    """
    Compute magnetic declination at geodetic coordinates using a ChaosMagPy model.

    Parameters
    ----------
    lat_gg : array-like
        Geodetic latitude(s) in degrees [-90, 90].
    lon : array-like
        Longitude(s) in degrees [-180, 180] or [0, 360].
    alt_km : array-like
        Altitude(s) above WGS84 ellipsoid in km.
    time_mjd2000 : float or array-like
        Time(s) as modified Julian date relative to 2000-01-01.
        Use cp.data_utils.mjd2000(year, month, day) to convert.
    model : chaosmagpy CHAOS or BaseModel instance
        A loaded ChaosMagPy model (e.g. from cp.load_CHAOS_shcfile()).

    Returns
    -------
    D : ndarray
        Magnetic declination in degrees. Positive eastward (magnetic north
        is east of geographic north), negative westward.

    Notes
    -----
    Declination is computed from the horizontal field components after
    converting geodetic coordinates to geocentric spherical coordinates.
    The model's time-dependent internal field is used (synth_values_tdep).
    """
    lat_gg       = np.atleast_1d(np.asarray(lat_gg, dtype=float))
    lon          = np.atleast_1d(np.asarray(lon,    dtype=float))
    alt_km       = np.atleast_1d(np.asarray(alt_km, dtype=float))
    time_mjd2000 = np.atleast_1d(np.asarray(time_mjd2000, dtype=float))

    colat_gg = 90.0 - lat_gg

    # Geodetic → geocentric (radius in km, colatitude in degrees)
    radius, colat_geo = gg_to_geo(alt_km, colat_gg)

    # Evaluate model field in geocentric spherical components
    B_radius, B_theta, B_phi = model.synth_values_tdep(
        time_mjd2000, radius, colat_geo, lon
    )
    
    # Horizontal components in geographic frame
    # B_theta points south → negate for northward; B_phi points east
    B_N = -B_theta
    B_E =  B_phi

    D = np.degrees(np.arctan2(B_E, B_N))

    return D


def nez_to_gg(B_N, B_E, B_Z, lat_gg, lon, alt_km, time_mjd2000, model):
    """
    Rotate magnetic field components from local magnetic NEZ to geodetic (GG)
    NED frame.

    The two-step transformation is:
        1. Rotate by declination D: magnetic NEZ → geographic NED (geocentric)
        2. Apply ellipsoidal correction:  geocentric NED → geodetic NED

    The actual signatures of the chaosmagpy conversion functions are:
        gg_to_geo(height, colat_gg, X=None, Z=None)
            → (radius, colat_geo) or (radius, colat_geo, B_radius, B_theta)
            where X is the northward and Z the downward geodetic component.

        geo_to_gg(radius, colat_geo, B_radius=None, B_theta=None)
            → (height, colat_gg) or (height, colat_gg, X, Z)
            where X is returned as northward and Z as downward geodetic component.

    Note: B_E / eastward is NOT passed to these functions — it is unaffected
    by the geocentric↔geodetic correction, which is a pure meridional rotation.

    Parameters
    ----------
    B_N : array-like
        Northward component in magnetic NEZ frame (nT).
    B_E : array-like
        Eastward component in magnetic NEZ frame (nT).
    B_Z : array-like
        Vertically downward component in magnetic NEZ frame (nT).
    lat_gg : array-like
        Geodetic latitude(s) in degrees [-90, 90].
    lon : array-like
        Longitude(s) in degrees.
    alt_km : array-like
        Altitude(s) above WGS84 ellipsoid in km.
    time_mjd2000 : float or array-like
        Time(s) as modified Julian date relative to 2000-01-01.
    model : chaosmagpy CHAOS or BaseModel instance
        A loaded ChaosMagPy model used to compute declination.

    Returns
    -------
    B_N_gg : ndarray
        Northward component in geodetic NED frame (nT).
    B_E_gg : ndarray
        Eastward component in geodetic NED frame (nT).
    B_Z_gg : ndarray
        Vertically downward component in geodetic NED frame (nT).
    """
    B_N          = np.atleast_1d(np.asarray(B_N,          dtype=float))
    B_E          = np.atleast_1d(np.asarray(B_E,          dtype=float))
    B_Z          = np.atleast_1d(np.asarray(B_Z,          dtype=float))
    lat_gg       = np.atleast_1d(np.asarray(lat_gg,       dtype=float))
    lon          = np.atleast_1d(np.asarray(lon,           dtype=float))
    alt_km       = np.atleast_1d(np.asarray(alt_km,       dtype=float))
    time_mjd2000 = np.atleast_1d(np.asarray(time_mjd2000, dtype=float))

    colat_gg = 90.0 - lat_gg

    # ------------------------------------------------------------------ #
    # Step 1: rotate magnetic NEZ → geocentric geographic NED by decl. D  #
    # ------------------------------------------------------------------ #
    D = compute_declination(lat_gg, lon, alt_km, time_mjd2000, model)
    D_rad = np.radians(D)

    B_N_geo =  B_N * np.cos(D_rad) - B_E * np.sin(D_rad)
    B_E_geo =  B_N * np.sin(D_rad) + B_E * np.cos(D_rad)
    B_Z_geo =  B_Z

    # ------------------------------------------------------------------ #
    # Step 2: geocentric geographic NED → geodetic NED                    #
    #                                                                      #
    # gg_to_geo takes geodetic (X=northward, Z=downward) and returns      #
    # geocentric (B_radius=upward, B_theta=southward).                    #
    # geo_to_gg is the inverse.                                           #
    # E is skipped entirely — it is invariant under this rotation.        #
    # ------------------------------------------------------------------ #

    # geocentric NED → geocentric spherical, pass N and Z through gg_to_geo
    # (even though the field is already geocentric, the rotation matrix from
    # gg_to_geo gives us the geocentric spherical components we need for geo_to_gg)
    radius, colat_geo, B_radius, B_theta = gg_to_geo(
        alt_km, colat_gg, B_N_geo, B_Z_geo   # X=northward, Z=downward
    )

    # geocentric spherical → geodetic NED
    _, _, B_N_gg, B_Z_gg = geo_to_gg(
        radius, colat_geo, B_radius, B_theta
    )

    # E is unchanged by the ellipsoidal correction
    B_E_gg = B_E_geo

    return B_N_gg, B_E_gg, B_Z_gg


import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def add_qd_gridlines(ax, apex_obj, 
                     qlat_lines=range(-10, 90, 10),
                     qlon_lines=range(-180, 181, 20),
                     n_points=500,
                     lat_color='blue', lon_color='blue',
                     linewidth=0.5, alpha=0.5, linestyle='--',
                     label_gridlines=True,   # ← new
                     extent=None):

    qlon_sweep = np.linspace(-180, 180, n_points)

    for qlat in qlat_lines:
        qlat_arr = np.full(n_points, float(qlat))
        geo_lat, geo_lon = apex_obj.convert(qlat_arr, qlon_sweep, 'qd', 'geo')

        splits = np.where(np.abs(np.diff(geo_lon)) > 90)[0] + 1
        for seg_lat, seg_lon in zip(np.split(geo_lat, splits), np.split(geo_lon, splits)):
            if len(seg_lat) < 2:
                continue
            ax.plot(seg_lon, seg_lat, transform=ccrs.PlateCarree(),
                    color=lat_color, linewidth=linewidth,
                    alpha=alpha, linestyle=linestyle, zorder=3)

        if label_gridlines and extent is not None:
            lon_min, lon_max, lat_min, lat_max = extent
            inside = (geo_lon >= lon_min) & (geo_lon <= lon_max) & \
                     (geo_lat >= lat_min) & (geo_lat <= lat_max)
            if inside.any():
                idx = np.where(inside)[0][0]
                ax.text(geo_lon[idx], geo_lat[idx], f'{int(qlat):+d}°',
                        transform=ccrs.PlateCarree(),
                        fontsize=7, color=lat_color, alpha=0.8,
                        ha='left', va='bottom', zorder=4)

    qlat_sweep = np.linspace(-90, 90, n_points)

    for qlon in qlon_lines:
        qlon_arr = np.full(n_points, float(qlon))
        geo_lat, geo_lon = apex_obj.convert(qlat_sweep, qlon_arr, 'qd', 'geo')

        splits = np.where(np.abs(np.diff(geo_lon)) > 90)[0] + 1
        for seg_lat, seg_lon in zip(np.split(geo_lat, splits), np.split(geo_lon, splits)):
            if len(seg_lat) < 2:
                continue
            ax.plot(seg_lon, seg_lat, transform=ccrs.PlateCarree(),
                    color=lon_color, linewidth=linewidth,
                    alpha=alpha, linestyle=linestyle, zorder=3)

        if label_gridlines and extent is not None:
            lon_min, lon_max, lat_min, lat_max = extent
            inside = (geo_lon >= lon_min) & (geo_lon <= lon_max) & \
                     (geo_lat >= lat_min) & (geo_lat <= lat_max)
            if inside.any():
                idx = np.where(inside)[0][0]
                ax.text(geo_lon[idx], geo_lat[idx], f'{int(qlon):+d}°',
                        transform=ccrs.PlateCarree(),
                        fontsize=7, color=lon_color, alpha=0.8,
                        ha='center', va='top', zorder=4)

def add_qd_gridlines_cs(csax, apex_obj,          # csax is a CSplot instance
                     qlat_lines=range(-10, 90, 10),
                     qlon_lines=range(-180, 181, 20),
                     n_points=500,
                     lat_color='blue', lon_color='blue',
                     linewidth=0.5, alpha=0.5, linestyle='--',
                     label_gridlines=True,
                     extent=None):

    qlon_sweep = np.linspace(-180, 180, n_points)

    for qlat in qlat_lines:
        qlat_arr = np.full(n_points, float(qlat))
        geo_lat, geo_lon = apex_obj.convert(qlat_arr, qlon_sweep, 'qd', 'geo')

        splits = np.where(np.abs(np.diff(geo_lon)) > 90)[0] + 1
        for seg_lat, seg_lon in zip(np.split(geo_lat, splits), np.split(geo_lon, splits)):
            if len(seg_lat) < 2:
                continue
            csax.plot(seg_lon, seg_lat,          # ← CSplot.plot(), no transform
                      lw=linewidth, color=lat_color,
                      alpha=alpha, linestyle=linestyle, zorder=3)

        if label_gridlines and extent is not None:
            lon_min, lon_max, lat_min, lat_max = extent
            inside = (geo_lon >= lon_min) & (geo_lon <= lon_max) & \
                     (geo_lat >= lat_min) & (geo_lat <= lat_max)
            if inside.any():
                idx = np.where(inside)[0][0]
                # labels go via the underlying ax since CSplot likely has no .text()
                csax.ax.text(geo_lon[idx], geo_lat[idx], f'{int(qlat):+d}°',
                             fontsize=7, color=lat_color, alpha=0.8,
                             ha='left', va='bottom', zorder=4)

    qlat_sweep = np.linspace(-90, 90, n_points)

    for qlon in qlon_lines:
        qlon_arr = np.full(n_points, float(qlon))
        geo_lat, geo_lon = apex_obj.convert(qlat_sweep, qlon_arr, 'qd', 'geo')

        splits = np.where(np.abs(np.diff(geo_lon)) > 90)[0] + 1
        for seg_lat, seg_lon in zip(np.split(geo_lat, splits), np.split(geo_lon, splits)):
            if len(seg_lat) < 2:
                continue
            csax.plot(seg_lon, seg_lat,          # ← CSplot.plot(), no transform
                      lw=linewidth, color=lon_color,
                      alpha=alpha, linestyle=linestyle, zorder=3)

        if label_gridlines and extent is not None:
            lon_min, lon_max, lat_min, lat_max = extent
            inside = (geo_lon >= lon_min) & (geo_lon <= lon_max) & \
                     (geo_lat >= lat_min) & (geo_lat <= lat_max)
            if inside.any():
                idx = np.where(inside)[0][0]
                csax.ax.text(geo_lon[idx], geo_lat[idx], f'{int(qlon):+d}°',
                             fontsize=7, color=lon_color, alpha=0.8,
                             ha='center', va='top', zorder=4)
                


""" Script to calculate the Xiong and Luhr auroral oval boundaries

    It can also be used to calculate unit vectors that are perpendicular
    to the auroral oval boundaries

    Kalle, Jan 2020


"""

def get_xiong_boundaries(By,Bz,v, resolution = 360, hemisphere = 'north', return_normal = False):
    """ return ellipses corresponding to inner and outer boundaries of auroral oval
        according to Xiong and Luhr (2014) "An empirical model of the auroral oval
        derived from CHAMP field-aligned current signatures - Part 2", Annales Geophysicae

        epsilon is the Newell epsilon parameter, using lags as specified by Xiong and Luhr

        the boundaries are ellipses. The return values are
        mlat_eq : magnetic latitudes  of the equatorward boundary at resolution points
        mlt_eq  : magnetic local time of the equatorward boundary at resolution points
        mlat_pw : magnetic latitudes  of the poleward    boundary at resolution points
        mlt_pw  : magnetic local time of the poleward    boundary at resolution points

        return_normal: in addition to the above, the e,n-components of inward normal vectors will be returned:
        ne_eq, nn_eq, ne_pw, nn_pw
    """
    ca = np.arctan2(By, Bz)
    epsilon = np.abs(v)**(4/3.) * np.sqrt(By**2 + Bz**2)**(2/3.) * (np.sin(ca/2)**(8))**(1/3.) / 1000 # Newell coupling           

    # get quadratic polynomials for each parameter, from tables 2 and 3 in the paper
    if hemisphere == 'north':
        # equatorward
        p_rx_eq = [-2.3836e-2,  9.5470e-1,  1.8861e1 ]
        p_ry_eq = [-2.9566e-2,  1.1504   ,  2.0562e1 ]
        p_x0_eq = [ 0        ,  2.7827e-2,  4.1263   ]
        p_y0_eq = [ 1.6569e-3, -2.8855e-2, -3.2637e-1]
        p_ph_eq = [ 0        ,  3.1147e-1, -3.1555   ]

        # poleward
        p_rx_pw = [-5.9729e-4,  2.6173e-1,  1.2813e1 ]
        p_ry_pw = [-2.9556e-2,  9.6759e-1,  9.5486   ]
        p_x0_pw = [ 0        , -2.5310e-1,  4.5175   ]
        p_y0_pw = [ 5.6513e-3, -1.5721e-1, -3.9319e-1]
        p_ph_pw = [ 0        ,  1.2831   , -8.5358   ]

    else: # assuming south
        # poleward
        p_rx_pw = [-1.3209e-2,  8.1597e-1,  1.8559e1 ]
        p_ry_pw = [-2.4605e-2,  1.0752e-1,  1.9549e1 ]
        p_x0_pw = [ 0        ,  4.4667e-2,  3.6946   ]
        p_y0_pw = [ 6.5985e-4, -1.1623e-2, -6.0436e-1]
        p_ph_pw = [ 0        , -1.0934e-1, -8.8836   ]

        # equatorward
        p_rx_eq = [-3.0559e-4,  2.8870e-1,  1.3251e1 ]
        p_ry_eq = [-2.4073e-2,  8.2006e-1,  1.1605e1 ]
        p_x0_eq = [ 0        , -2.1674e-1,  4.2526   ]
        p_y0_eq = [ 7.0729e-4, -2.4479e-1, -1.1330   ]
        p_ph_eq = [ 0        , -1.5508   ,  3.7050   ]


    rx_pw = p_rx_pw[0] * epsilon**2 + p_rx_pw[1] * epsilon + p_rx_pw[2]
    ry_pw = p_ry_pw[0] * epsilon**2 + p_ry_pw[1] * epsilon + p_ry_pw[2]
    x0_pw = p_x0_pw[0] * epsilon**2 + p_x0_pw[1] * epsilon + p_x0_pw[2]
    y0_pw = p_y0_pw[0] * epsilon**2 + p_y0_pw[1] * epsilon + p_y0_pw[2]
    ph_pw = p_ph_pw[0] * epsilon**2 + p_ph_pw[1] * epsilon + p_ph_pw[2]
    ph_pw = ph_pw * np.pi/180

    rx_eq = p_rx_eq[0] * epsilon**2 + p_rx_eq[1] * epsilon + p_rx_eq[2]
    ry_eq = p_ry_eq[0] * epsilon**2 + p_ry_eq[1] * epsilon + p_ry_eq[2]
    x0_eq = p_x0_eq[0] * epsilon**2 + p_x0_eq[1] * epsilon + p_x0_eq[2]
    y0_eq = p_y0_eq[0] * epsilon**2 + p_y0_eq[1] * epsilon + p_y0_eq[2]
    ph_eq = p_ph_eq[0] * epsilon**2 + p_ph_eq[1] * epsilon + p_ph_eq[2]
    ph_eq = ph_eq * np.pi/180

    # define ellipses in canonical form (axes aligned with coordinates):
    a = np.linspace(0, 2*np.pi, resolution)
    x_eq_c = rx_eq*np.cos(a)
    y_eq_c = ry_eq*np.sin(a)
    x_pw_c = rx_pw*np.cos(a)
    y_pw_c = ry_pw*np.sin(a)

    # define rotation matrix to rotate
    R_eq = np.array([[np.cos(ph_eq), np.sin(ph_eq)], [-np.sin(ph_eq), np.cos(ph_eq)]])
    R_pw = np.array([[np.cos(ph_pw), np.sin(ph_pw)], [-np.sin(ph_pw), np.cos(ph_pw)]])

    # rotate:
    x_eq, y_eq = R_eq.dot(np.vstack((x_eq_c, y_eq_c)))
    x_pw, y_pw = R_pw.dot(np.vstack((x_pw_c, y_pw_c)))

    # and translate:
    x_eq = x_eq + x0_eq
    y_eq = y_eq + y0_eq

    x_pw = x_pw + x0_pw
    y_pw = y_pw + y0_pw

    # and calculate mlat and mlt:
    mlt_eq  = np.arctan2(y_eq, x_eq) * 12/np.pi
    mlt_pw  = np.arctan2(y_pw, x_pw) * 12/np.pi
    mlat_eq = 90 - np.sqrt(y_eq**2 + x_eq**2)
    mlat_pw = 90 - np.sqrt(y_pw**2 + x_pw**2)


    if return_normal:
        # east and north components of normals in canonical form
        e_eq = -(rx_eq - ry_eq) * np.cos(a) * np.sin(a)       / np.sqrt(rx_eq**2 * np.cos(a)**2 + ry_eq**2 * np.sin(a)**2)
        n_eq =  (rx_eq * np.cos(a)**2 + ry_eq * np.sin(a)**2) / np.sqrt(rx_eq**2 * np.cos(a)**2 + ry_eq**2 * np.sin(a)**2)

        e_pw = -(rx_pw - ry_pw) * np.cos(a) * np.sin(a)       / np.sqrt(rx_pw**2 * np.cos(a)**2 + ry_pw**2 * np.sin(a)**2)
        n_pw =  (rx_pw * np.cos(a)**2 + ry_pw * np.sin(a)**2) / np.sqrt(rx_pw**2 * np.cos(a)**2 + ry_pw**2 * np.sin(a)**2)


        # calculate the cartesian components, and rotate into correct system
        X_eq = -e_eq * np.sin(a) + n_eq * np.cos(a)
        Y_eq =  e_eq * np.cos(a) + n_eq * np.sin(a)
        X_pw = -e_pw * np.sin(a) + n_pw * np.cos(a)
        Y_pw =  e_pw * np.cos(a) + n_pw * np.sin(a)

        X_eq, Y_eq = R_eq.dot(np.vstack((X_eq, Y_eq)))
        X_pw, Y_pw = R_pw.dot(np.vstack((X_pw, Y_pw)))

        # calculate the e, n - components from the rotated cartesian components
        e_eq =  np.sin(mlt_eq * np.pi/12)  * X_eq - np.cos(mlt_eq * np.pi/12) * Y_eq
        n_eq =  np.cos(mlt_eq * np.pi/12)  * X_eq + np.sin(mlt_eq * np.pi/12) * Y_eq
        e_pw =  np.sin(mlt_pw * np.pi/12)  * X_pw - np.cos(mlt_pw * np.pi/12) * Y_pw
        n_pw =  np.cos(mlt_pw * np.pi/12)  * X_pw + np.sin(mlt_pw * np.pi/12) * Y_pw

        return mlat_eq, mlt_eq, mlat_pw, mlt_pw, e_eq, n_eq, e_pw, n_pw
    else:
        return mlat_eq, mlt_eq, mlat_pw, mlt_pw


