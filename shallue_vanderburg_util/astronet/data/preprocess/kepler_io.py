# Copyright 2018 The Exoplanet ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for reading Kepler data."""

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
#
import os.path
from os import path
from astropy.io import fits
import numpy as np
#
#
#import numpy as np
#import scipy.interpolate
#from six.moves import range  # pylint:disable=redefined-builtin


def phase_fold_time(time, period, t0):
  """Creates a phase-folded time vector.

  result[i] is the unique number in [-period / 2, period / 2)
  such that result[i] = time[i] - t0 + k_i * period, for some integer k_i.

  Args:
    time: 1D numpy array of time values.
    period: A positive real scalar; the period to fold over.
    t0: The center of the resulting folded vector; this value is mapped to 0.

  Returns:
    A 1D numpy array.
  """
  half_period = period / 2
  result = np.mod(time + (half_period - t0), period)
  result -= half_period
  return result


def split(all_time, all_flux, gap_width=0.75):
  """Splits a light curve on discontinuities (gaps).

  This function accepts a light curve that is either a single segment, or is
  piecewise defined (e.g. split by quarter breaks or gaps in the in the data).

  Args:
    all_time: Numpy array or sequence of numpy arrays; each is a sequence of
      time values.
    all_flux: Numpy array or sequence of numpy arrays; each is a sequence of
      flux values of the corresponding time array.
    gap_width: Minimum gap size (in time units) for a split.

  Returns:
    out_time: List of numpy arrays; the split time arrays.
    out_flux: List of numpy arrays; the split flux arrays.
  """
  # Handle single-segment inputs.
  if isinstance(all_time, np.ndarray) and all_time.ndim == 1:
    all_time = [all_time]
    all_flux = [all_flux]

  out_time = []
  out_flux = []
  for time, flux in zip(all_time, all_flux):
    start = 0
    for end in range(1, len(time) + 1):
      # Choose the largest endpoint such that time[start:end] has no gaps.
      if end == len(time) or time[end] - time[end - 1] > gap_width:
        out_time.append(time[start:end])
        out_flux.append(flux[start:end])
        start = end

  return out_time, out_flux


def remove_events(all_time,
                  all_flux,
                  events,
                  width_factor=1.0,
                  include_empty_segments=True):
  """Removes events from a light curve.

  This function accepts either a single-segment or piecewise-defined light
  curve (e.g. one that is split by quarter breaks or gaps in the in the data).

  Args:
    all_time: Numpy array or sequence of numpy arrays; each is a sequence of
      time values.
    all_flux: Numpy array or sequence of numpy arrays; each is a sequence of
      flux values of the corresponding time array.
    events: List of Event objects to remove.
    width_factor: Fractional multiplier of the duration of each event to remove.
    include_empty_segments: Whether to include empty segments in the output.

  Returns:
    output_time: Numpy array or list of numpy arrays; the time arrays with
        events removed.
    output_flux: Numpy array or list of numpy arrays; the flux arrays with
        events removed.
  """
  # Handle single-segment inputs.
  if isinstance(all_time, np.ndarray) and all_time.ndim == 1:
    all_time = [all_time]
    all_flux = [all_flux]
    single_segment = True
  else:
    single_segment = False

  output_time = []
  output_flux = []
  for time, flux in zip(all_time, all_flux):
    mask = np.ones_like(time, dtype=np.bool)
    for event in events:
      transit_dist = np.abs(phase_fold_time(time, event.period, event.t0))
      mask = np.logical_and(mask,
                            transit_dist > 0.5 * width_factor * event.duration)

    if single_segment:
      output_time = time[mask]
      output_flux = flux[mask]
    elif include_empty_segments or np.any(mask):
      output_time.append(time[mask])
      output_flux.append(flux[mask])

  return output_time, output_flux


def interpolate_missing_time(time, cadence_no=None, fill_value="extrapolate"):
  """Interpolates missing (NaN or Inf) time values.

  Args:
    time: A numpy array of monotonically increasing values, with missing values
      denoted by NaN or Inf.
    cadence_no: Optional numpy array of cadence numbers corresponding to the
      time values. If not provided, missing time values are assumed to be evenly
      spaced between present time values.
    fill_value: Specifies how missing time values should be treated at the
      beginning and end of the array. See scipy.interpolate.interp1d.

  Returns:
    A numpy array of the same length as the input time array, with NaN/Inf
    values replaced with interpolated values.

  Raises:
    ValueError: If fewer than 2 values of time are finite.
  """
  if cadence_no is None:
    cadence_no = np.arange(len(time))

  is_finite = np.isfinite(time)
  num_finite = np.sum(is_finite)
  if num_finite < 2:
    raise ValueError(
        "Cannot interpolate time with fewer than 2 finite values. Got "
        "len(time) = {} with {} finite values.".format(len(time), num_finite))

  interpolate_fn = scipy.interpolate.interp1d(
      cadence_no[is_finite],
      time[is_finite],
      copy=False,
      bounds_error=False,
      fill_value=fill_value,
      assume_sorted=True)

  return interpolate_fn(cadence_no)


def interpolate_masked_spline(all_time, all_masked_time, all_masked_spline):
  """Linearly interpolates spline values across masked points.

  Args:
    all_time: List of numpy arrays; each is a sequence of time values.
    all_masked_time: List of numpy arrays; each is a sequence of time values
      with some values missing (masked).
    all_masked_spline: List of numpy arrays; the masked spline values
      corresponding to all_masked_time.

  Returns:
    interp_spline: List of numpy arrays; each is the masked spline with missing
        points linearly interpolated.
  """
  interp_spline = []
  for time, masked_time, masked_spline in zip(all_time, all_masked_time,
                                              all_masked_spline):
    if masked_time.size:
      interp_spline.append(np.interp(time, masked_time, masked_spline))
    else:
      interp_spline.append(np.array([np.nan] * len(time)))
  return interp_spline


def reshard_arrays(xs, ys):
  """Reshards arrays in xs to match the lengths of arrays in ys.

  Args:
    xs: List of 1d numpy arrays with the same total length as ys.
    ys: List of 1d numpy arrays with the same total length as xs.

  Returns:
    A list of numpy arrays containing the same elements as xs, in the same
    order, but with array lengths matching the pairwise array in ys.

  Raises:
    ValueError: If xs and ys do not have the same total length.
  """
  # Compute indices of boundaries between segments of ys, plus the end boundary.
  boundaries = np.cumsum([len(y) for y in ys])
  concat_x = np.concatenate(xs)
  if len(concat_x) != boundaries[-1]:
    raise ValueError(
        "xs and ys do not have the same total length ({} vs. {}).".format(
            len(concat_x), boundaries[-1]))
  boundaries = boundaries[:-1]  # Remove exclusive end boundary.
  return np.split(concat_x, boundaries)


def uniform_cadence_light_curve(cadence_no, time, flux):
  """Combines data into a single light curve with uniform cadence numbers.

  Args:
    cadence_no: numpy array; the cadence numbers of the light curve.
    time: numpy array; the time values of the light curve.
    flux: numpy array; the flux values of the light curve.

  Returns:
    cadence_no: numpy array; the cadence numbers of the light curve with no
      gaps. It starts and ends at the minimum and maximum cadence numbers in the
      input light curve, respectively.
    time: numpy array; the time values of the light curve. Missing data points
      have value zero and correspond to a False value in the mask.
    flux: numpy array; the time values of the light curve. Missing data points
      have value zero and correspond to a False value in the mask.
    mask: Boolean numpy array; False indicates missing data points, where
      missing data points are those that have no corresponding cadence number in
      the input or those where at least one of the cadence number, time value,
      or flux value is NaN/Inf.

  Raises:
    ValueError: If there are duplicate cadence numbers in the input.
  """
  min_cadence_no = np.min(cadence_no)
  max_cadence_no = np.max(cadence_no)

  out_cadence_no = np.arange(
      min_cadence_no, max_cadence_no + 1, dtype=cadence_no.dtype)
  out_time = np.zeros_like(out_cadence_no, dtype=time.dtype)
  out_flux = np.zeros_like(out_cadence_no, dtype=flux.dtype)
  out_mask = np.zeros_like(out_cadence_no, dtype=np.bool)

  for c, t, f in zip(cadence_no, time, flux):
    if np.isfinite(c) and np.isfinite(t) and np.isfinite(f):
      i = int(c - min_cadence_no)
      if out_mask[i]:
        raise ValueError("Duplicate cadence number: {}".format(c))
      out_time[i] = t
      out_flux[i] = f
      out_mask[i] = True

  return out_cadence_no, out_time, out_flux, out_mask


def count_transit_points(time, event):
  """Computes the number of points in each transit of a given event.

  Args:
    time: Sorted numpy array of time values.
    event: An Event object.

  Returns:
    A numpy array containing the number of time points "in transit" for each
    transit occurring between the first and last time values.

  Raises:
    ValueError: If there are more than 10**6 transits.
  """
  t_min = np.min(time)
  t_max = np.max(time)

  # Tiny periods or erroneous time values could make this loop take forever.
  if (t_max - t_min) / event.period > 10**6:
    raise ValueError(
        "Too many transits! Time range is [{:.4f}, {:.4f}] and period is "
        "{:.4e}.".format(t_min, t_max, event.period))

  # Make sure t0 is in [t_min, t_min + period).
  t0 = np.mod(event.t0 - t_min, event.period) + t_min

  # Prepare loop variables.
  points_in_transit = []
  i, j = 0, 0

  for transit_midpoint in np.arange(t0, t_max, event.period):
    transit_begin = transit_midpoint - event.duration / 2
    transit_end = transit_midpoint + event.duration / 2

    # Move time[i] to the first point >= transit_begin.
    while time[i] < transit_begin:
      # transit_begin is guaranteed to be < np.max(t) (provided duration >= 0).
      # Therefore, i cannot go out of range.
      i += 1

    # Move time[j] to the first point > transit_end.
    while time[j] <= transit_end:
      j += 1
      # j went out of range. We're finished.
      if j >= len(time):
        break

    # The points in the current transit duration are precisely time[i:j].
    # Since j is an exclusive index, there are exactly j-i points in transit.
    points_in_transit.append(j - i)

  return np.array(points_in_transit)


# Quarter index to filename prefix for long cadence Kepler data.
# Reference: https://archive.stsci.edu/kepler/software/get_kepler.py
LONG_CADENCE_QUARTER_PREFIXES = {
    0: ["2009131105131"],
    1: ["2009166043257"],
    2: ["2009259160929"],
    3: ["2009350155506"],
    4: ["2010078095331", "2010009091648"],
    5: ["2010174085026"],
    6: ["2010265121752"],
    7: ["2010355172524"],
    8: ["2011073133259"],
    9: ["2011177032512"],
    10: ["2011271113734"],
    11: ["2012004120508"],
    12: ["2012088054726"],
    13: ["2012179063303"],
    14: ["2012277125453"],
    15: ["2013011073258"],
    16: ["2013098041711"],
    17: ["2013131215648"]
}

# Quarter index to filename prefix for short cadence Kepler data.
# Reference: https://archive.stsci.edu/kepler/software/get_kepler.py
SHORT_CADENCE_QUARTER_PREFIXES = {
    0: ["2009131110544"],
    1: ["2009166044711"],
    2: ["2009201121230", "2009231120729", "2009259162342"],
    3: ["2009291181958", "2009322144938", "2009350160919"],
    4: ["2010009094841", "2010019161129", "2010049094358", "2010078100744"],
    5: ["2010111051353", "2010140023957", "2010174090439"],
    6: ["2010203174610", "2010234115140", "2010265121752"],
    7: ["2010296114515", "2010326094124", "2010355172524"],
    8: ["2011024051157", "2011053090032", "2011073133259"],
    9: ["2011116030358", "2011145075126", "2011177032512"],
    10: ["2011208035123", "2011240104155", "2011271113734"],
    11: ["2011303113607", "2011334093404", "2012004120508"],
    12: ["2012032013838", "2012060035710", "2012088054726"],
    13: ["2012121044856", "2012151031540", "2012179063303"],
    14: ["2012211050319", "2012242122129", "2012277125453"],
    15: ["2012310112549", "2012341132017", "2013011073258"],
    16: ["2013017113907", "2013065031647", "2013098041711"],
    17: ["2013121191144", "2013131215648"]
}

# Quarter order for different scrambling procedures.
# Page 9: https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20170009549.pdf.
SIMULATED_DATA_SCRAMBLE_ORDERS = {
    "SCR1": [0, 13, 14, 15, 16, 9, 10, 11, 12, 5, 6, 7, 8, 1, 2, 3, 4, 17],
    "SCR2": [0, 1, 2, 3, 4, 13, 14, 15, 16, 9, 10, 11, 12, 5, 6, 7, 8, 17],
    "SCR3": [0, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 17],
}


def kepler_filenames(base_dir,
                     kep_id,
                     long_cadence=True,
                     quarters=None,
                     injected_group=None,
                     check_existence=True):
  """Returns the light curve filenames for a Kepler target star.

  This function assumes the directory structure of the Mikulski Archive for
  Space Telescopes (http://archive.stsci.edu/pub/kepler/lightcurves).
  Specifically, the filenames for a particular Kepler target star have the
  following format:

    ${kep_id:0:4}/${kep_id}/kplr${kep_id}-${quarter_prefix}_${type}.fits,

  where:
    kep_id is the Kepler id left-padded with zeros to length 9;
    quarter_prefix is the filename quarter prefix;
    type is one of "llc" (long cadence light curve) or "slc" (short cadence
        light curve).

  Args:
    base_dir: Base directory containing Kepler data.
    kep_id: Id of the Kepler target star. May be an int or a possibly zero-
      padded string.
    long_cadence: Whether to read a long cadence (~29.4 min / measurement) light
      curve as opposed to a short cadence (~1 min / measurement) light curve.
    quarters: Optional list of integers in [0, 17]; the quarters of the Kepler
      mission to return.
    injected_group: Optional string indicating injected light curves. One of
      "inj1", "inj2", "inj3".
    check_existence: If True, only return filenames corresponding to files that
      exist (not all stars have data for all quarters).

  Returns:
    A list of filenames.
  """
  # Pad the Kepler id with zeros to length 9.
  kep_id = "{:09d}".format(int(kep_id))

  quarter_prefixes, cadence_suffix = ((LONG_CADENCE_QUARTER_PREFIXES, "llc")
                                      if long_cadence else
                                      (SHORT_CADENCE_QUARTER_PREFIXES, "slc"))

  if quarters is None:
    quarters = quarter_prefixes.keys()

  quarters = sorted(quarters)  # Sort quarters chronologically.

  filenames = []
  base_dir = os.path.join(base_dir, kep_id[0:4], kep_id)
  for quarter in quarters:
    for quarter_prefix in quarter_prefixes[quarter]:
      if injected_group:
        base_name = "kplr{}-{}_INJECTED-{}_{}.fits".format(
            kep_id, quarter_prefix, injected_group, cadence_suffix)
      else:
        base_name = "kplr{}-{}_{}.fits".format(kep_id, quarter_prefix,
                                               cadence_suffix)
      filename = os.path.join(base_dir, base_name)
      # Not all stars have data for all quarters.
      if not check_existence or path.exists(filename):
        filenames.append(filename)

  return filenames


def scramble_light_curve(all_time, all_flux, all_quarters, scramble_type):
  """Scrambles a light curve according to a given scrambling procedure.

  Args:
    all_time: List holding arrays of time values, each containing a quarter of
      time data.
    all_flux: List holding arrays of flux values, each containing a quarter of
      flux data.
    all_quarters: List of integers specifying which quarters are present in
      the light curve (max is 18: Q0...Q17).
    scramble_type: String specifying the scramble order, one of {'SCR1', 'SCR2',
      'SCR3'}.

  Returns:
    scr_flux: Scrambled flux values; the same list as the input flux in another
      order.
    scr_time: Time values, re-partitioned to match sizes of the scr_flux lists.
  """
  order = SIMULATED_DATA_SCRAMBLE_ORDERS[scramble_type]
  scr_flux = []
  for quarter in order:
    # Ignore missing quarters in the scramble order.
    if quarter in all_quarters:
      scr_flux.append(all_flux[all_quarters.index(quarter)])

  scr_time = util.reshard_arrays(all_time, scr_flux)

  return scr_time, scr_flux


def read_kepler_light_curve(filenames,
                            light_curve_extension="LIGHTCURVE",
                            scramble_type=None,
                            interpolate_missing_time=False,
                            invert=False):
  """Reads time and flux measurements for a Kepler target star.

  Args:
    filenames: A list of .fits files containing time and flux measurements.
    light_curve_extension: Name of the HDU 1 extension containing light curves.
    scramble_type: What scrambling procedure to use: 'SCR1', 'SCR2', or 'SCR3'
      (pg 9: https://exoplanetarchive.ipac.caltech.edu/docs/KSCI-19114-002.pdf).
    interpolate_missing_time: Whether to interpolate missing (NaN) time values.
      This should only affect the output if scramble_type is specified (NaN time
      values typically come with NaN flux values, which are removed anyway, but
      scrambling decouples NaN time values from NaN flux values).
    invert: Whether to reflect flux values around the median flux value. This is
      performed separately for each .fits file.

  Returns:
    all_time: A list of numpy arrays; the time values of the light curve.
    all_flux: A list of numpy arrays; the flux values of the light curve.
  """
  all_time = []
  all_flux = []
  all_quarters = []

  for filename in filenames:
    with fits.open(open(filename, "rb")) as hdu_list:
      quarter = hdu_list["PRIMARY"].header["QUARTER"]
      light_curve = hdu_list[light_curve_extension].data

    time = light_curve.TIME
    flux = light_curve.PDCSAP_FLUX
    if not time.size:
      continue  # No data.

    # Possibly interpolate missing time values.
    if interpolate_missing_time:
      time = util.interpolate_missing_time(time, light_curve.CADENCENO)

    all_time.append(time)
    all_flux.append(flux)
    all_quarters.append(quarter)

  if scramble_type:
    all_time, all_flux = scramble_light_curve(all_time, all_flux, all_quarters,
                                              scramble_type)

  # Remove timestamps with NaN time or flux values.
  for i, (time, flux) in enumerate(zip(all_time, all_flux)):
    flux_and_time_finite = np.logical_and(np.isfinite(flux), np.isfinite(time))
    all_time[i] = time[flux_and_time_finite]
    all_flux[i] = flux[flux_and_time_finite]

  # Possibly reflect each file's flux about its median value.
  if invert:
    for flux in all_flux:
      flux -= 2 * np.median(flux)
      flux *= -1

  return all_time, all_flux
