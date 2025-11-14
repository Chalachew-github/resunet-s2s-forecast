# ============================================================
#Utilities for ResUNet S2S Forecast Framework
#============================================================
import os
import sys
import time
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
xr.set_options(display_style='text')
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback

from datetime import datetime
import climetlab_s2s_ai_challenge
import climetlab as cml

cache_path = '/training-input'
# ============================================================
# PATH UTILITIES
# ============================================================
def get_paths(path):
    """
    get paths for data, predictions, trained models and result.csv based on the current folder structure
    adapt the paths below according to your folder structure

    Parameters
    ----------
    path: string, indicates what folder structure is present

    Returns
    -------
    cache_path: string, path to data provided by S2S AI Challenge
    path_add_vars: string, path to additional data
    path_model: string, path to folder where trained models should be saved
    path_pred: string, path to folder where predictions should be saved
    path_results: string, path to folder where all results including results.csv are saved
    """

    # paths to load forecasts and observations
    if path == 'server':
        cache_path = '/training-input'
        path_add_vars = '/training-input'
        path_add_vars2 = '/test-input'
    elif path == 'local':
        cache_path = '/training-input'
        path_add_vars = '/training-input'
        path_add_vars2 = '/test-input'

    # path to models
    if path == 'server':
        path_model = '/results/trained_models/'
    else:
        path_model = '/results/trained_models/'

    # path to save predictions
    if path == 'server':
        path_pred = '/results/predictions/'
    else:
        path_pred = '/results/predictions/'

    # path to results.csv
    if path == 'server':
        path_results = '/results/'
    else:
        path_results = '/results/'

    return cache_path, path_add_vars, path_model, path_pred, path_results, path_add_vars2


# ============================================================
# S2S AI CHALLENGE HELPERS
# ============================================================
""" taken from the S2S AI Challenge
    https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge-template/-/tree/master/notebooks
    Outcomes of the WMO Prize Challenge to Improve Sub-Seasonal
    to Seasonal Predictions Using Artificial Intelligence, Vitart et al. 2022
"""




def download(varlist_forecast=['tp','t2m'],
             center_list=['ecwmf'],
             forecast_dataset_labels=['hindcast-input','forecast-input'],
             obs_dataset_labels=['hindcast-like-observations','forecast-like-observations'],
             varlist_observations=['t2m','tp'],
             benchmark=True,
             format='netcdf'
            ):
    """Download files with climetlab to cache_path. Set cache_path:
    cml.settings.set("cache-directory", cache_path)
    """
    if isinstance(center_list, str):
        center_list = [center_list]
    if isinstance(varlist_forecast, str):
        varlist_forecast = [varlist_forecast]

    dates = xr.cftime_range(start='20200102',freq='7D', periods=53).strftime('%Y%m%d').to_list()
    
    if forecast_dataset_labels:
        print(f'Downloads variables {varlist_forecast} from datasets {forecast_dataset_labels} from center {center_list} in {format} format.')
        for center in center_list:
            for ds in forecast_dataset_labels:
                for parameter in varlist_forecast: 
                    try:
                        cml.load_dataset(f"s2s-ai-challenge-{ds}", origin=center, parameter=varlist_forecast, format=format).to_xarray()
                    except:
                        pass
    if obs_dataset_labels:
        print(f'Downloads variables tp and t2m from datasets {obs_dataset_labels} netcdf format. Additionally downloads raw t2m and pr observations with a time dimension.')
        try:
            for ds in obs_dataset_labels:
                for parameter in varlist_observations:
                    cml.load_dataset(f"s2s-ai-challenge-{ds}", date=dates, parameter=parameter).to_xarray()
        except:
            pass
        # raw
        cml.load_dataset(f"s2s-ai-challenge-observations", parameter=varlist_observations).to_xarray()
    if benchmark:
        cml.load_dataset("s2s-ai-challenge-test-output-benchmark", parameter=['tp','t2m']).to_xarray()
    print('finished')
    return


def add_valid_time_from_forecast_reference_time_and_lead_time(forecast, init_dim='forecast_time'):
    """Creates valid_time(forecast_time, lead_time).
    
    lead_time: pd.Timedelta
    forecast_time: datetime
    """
    times = xr.concat(
        [
            xr.DataArray(
                forecast[init_dim] + lead,
                dims=init_dim,
                coords={init_dim: forecast[init_dim]},
            )
            for lead in forecast.lead_time
        ],
        dim="lead_time",
        join="inner",
        compat="broadcast_equals",
    )
    forecast = forecast.assign_coords(valid_time=times)
    return forecast


def aggregate_biweekly(da):
    """
    Aggregate initialized S2S forecasts biweekly for xr.DataArrays.
    Use ds.map(aggregate_biweekly) for xr.Datasets.
    
    Applies to the ECMWF S2S data model: https://confluence.ecmwf.int/display/S2S/Parameters
    """
    # biweekly averaging
    w1 = [pd.Timedelta(f'{i} d') for i in range(1, 7)]
    w1 = xr.DataArray(w1, dims='lead_time', coords={'lead_time': w1})

    w2 = [pd.Timedelta(f'{i} d') for i in range(7, 14)]
    w2 = xr.DataArray(w2, dims='lead_time', coords={'lead_time': w2})

    w3 = [pd.Timedelta(f'{i} d') for i in range(14, 21)]
    w3 = xr.DataArray(w3, dims='lead_time', coords={'lead_time': w3})

    w4 = [pd.Timedelta(f'{i} d') for i in range(21, 28)]
    w4 = xr.DataArray(w4, dims='lead_time', coords={'lead_time': w4})

    w5 = [pd.Timedelta(f'{i} d') for i in range(28,35)]
    w5 = xr.DataArray(w5,dims='lead_time', coords={'lead_time':w5})
    
    w6 = [pd.Timedelta(f'{i} d') for i in range(35,42)]
    w6 = xr.DataArray(w6,dims='lead_time', coords={'lead_time':w6})
    
    biweekly_lead = [pd.Timedelta(f"{i} d") for i in [1, 7, 14, 21, 28, 35]] # take first day of biweekly average as new coordinate

    v = da.name
    if climetlab_s2s_ai_challenge.CF_CELL_METHODS[v] == 'sum': # biweekly difference for sum variables: tp and ttr
        d1 = da.sel(lead_time=pd.Timedelta("7 d")) - da.sel(lead_time=pd.Timedelta("1 d")) # tp from day 14 to day 27
        d2 = da.sel(lead_time=pd.Timedelta("14 d")) - da.sel(lead_time=pd.Timedelta("7 d")) # tp from day 28 to day 42
        d3 = da.sel(lead_time=pd.Timedelta("21 d")) - da.sel(lead_time=pd.Timedelta("14 d")) # tp from day 14 to day 27
        d4 = da.sel(lead_time=pd.Timedelta("28 d")) - da.sel(lead_time=pd.Timedelta("21 d")) # tp from day 28 to day 42
        d5 = da.sel(lead_time=pd.Timedelta("35 d")) - da.sel(lead_time=pd.Timedelta("28 d")) # tp from day 14 to day 27
        d6 = da.sel(lead_time=pd.Timedelta("42 d")) - da.sel(lead_time=pd.Timedelta("35 d")) # tp from day 28 to day 42
        da_biweekly = xr.concat([d1,d2,d3,d4,d5,d6],'lead_time').assign_coords(lead_time=biweekly_lead)
    else: # t2m, see climetlab_s2s_ai_challenge.CF_CELL_METHODS # biweekly: mean [day 14, day 27]
        d1 = da.sel(lead_time=w1).mean('lead_time')
        d2 = da.sel(lead_time=w2).mean('lead_time')
        d3 = da.sel(lead_time=w3).mean('lead_time')
        d4 = da.sel(lead_time=w4).mean('lead_time')
        d5 = da.sel(lead_time=w5).mean('lead_time')
        d6 = da.sel(lead_time=w6).mean('lead_time')
        da_biweekly = xr.concat([d1,d2,d3,d4,d5,d6],'lead_time').assign_coords(lead_time=biweekly_lead)
    
    da_biweekly = add_valid_time_from_forecast_reference_time_and_lead_time(da_biweekly)
    da_biweekly['lead_time'].attrs = {'long_name':'forecast_period', 'description': 'Forecast period is the time interval between the forecast reference time and the validity time.',
                         'aggregate': 'The pd.Timedelta corresponds to the first day of a biweekly aggregate.',
                         'week1_t2m': 'mean[day 1, 7]',
                         'week2_t2m': 'mean[day 7, 14]',
                         'week3_t2m': 'mean[day 14, 21]',
                         'week4_t2m': 'mean[day 21, 28]',
                         'week5_t2m': 'mean[day 28, 35]',
                         'week6_t2m': 'mean[day 35, 42]',
                         'week1_tp': 'day 7 minus day 1',
                         'week2_tp': 'day 14 minus day 7',
                         'week3_tp': 'day 21 minus day 14',
                         'week4_tp': 'day 28 minus day 21',
                         'week5_tp': 'day 35 minus day 28',
                         'week6_tp': 'day 42 minus day 35'}
    return da_biweekly


def ensure_attributes(da, biweekly=False):
    """Ensure that coordinates and variables have proper attributes. Set biweekly==True to set special comments for the biweely aggregates."""
    #template = cml.load_dataset('s2s-ai-challenge-test-input',parameter='t2m', origin='ecmwf', format='netcdf', date='20200102').to_xarray()
    template = xr.open_dataset('/training-input/ecmwf/t2m/ecmwf-hindcast-t2m-20200102.nc')

    for c in da.coords:
        if c in template.coords:
            da.coords[c].attrs.update(template.coords[c].attrs)
    
    if 'valid_time' in da.coords:
        da['valid_time'].attrs.update({'long_name': 'validity time',
                                     'standard_name': 'time',
                                     'description': 'time for which the forecast is valid',
                                     'calculate':'forecast_time + lead_time'})
    if 'forecast_time' in da.coords:
        da['forecast_time'].attrs.update({'long_name' : 'initial time of forecast', 'standard_name': 'forecast_reference_time',
                                      'description':'The forecast reference time in NWP is the "data time", the time of the analysis from which the forecast was made. It is not the time for which the forecast is valid.'})
    # fix tp
    if da.name == 'tp':
        da.attrs['units'] = 'kg m-2'
    if biweekly:
        da['lead_time'].attrs.update({'standard_name':'forecast_period', 'long_name': 'lead time',
                                      'description': 'Forecast period is the time interval between the forecast reference time and the validity time.',
                         'aggregate': 'The pd.Timedelta corresponds to the first day of a biweekly aggregate.',
                         'week1_t2m': 'mean[day 1, 7]',
                         'week2_t2m': 'mean[day 7, 14]',
                         'week3_t2m': 'mean[day 14, 21]',
                         'week4_t2m': 'mean[day 21, 28]',
                         'week5_t2m': 'mean[day 28, 35]',
                         'week6_t2m': 'mean[day 35, 42]',
                         'week1_tp': 'day 7 minus day 1',
                         'week2_tp': 'day 14 minus day 7',
                         'week3_tp': 'day 21 minus day 14',
                         'week4_tp': 'day 28 minus day 21',
                         'week5_tp': 'day 35 minus day 28',
                         'week6_tp': 'day 42 minus day 35'})
        if da.name == 'tp':
            da.attrs.update({'aggregate_week1': '7 days minus 1 days',
                      'aggregate_week2': '14 days minus 7 days',
                      'aggregate_week3': '21 days minus 14 days',
                      'aggregate_week4': '28 days minus 21 days',
                      'aggregate_week5': '35 days minus 28 days',
                      'aggregate_week6': '42 days minus 35 days',
                      'description': 'https://confluence.ecmwf.int/display/S2S/S2S+Total+Precipitation'})
        if da.name == 't2m':
            da.attrs.update({'aggregate_week1': 'mean[7 days, 1 days]',
                      'aggregate_week2': 'mean[14 days, 7 days]',
                      'aggregate_week3': 'mean[21 days, 14 days]',
                      'aggregate_week4': 'mean[28 days, 21 days]',
                      'aggregate_week5': 'mean[35 days, 28 days]',
                      'aggregate_week6': 'mean[42 days, 35 days]',
                      'variable_before_categorization': 'https://confluence.ecmwf.int/display/S2S/S2S+Surface+Air+Temperature'})
    return da


def add_year_week_coords(ds):
    if 'week' not in ds.coords and 'year' not in ds.coords:
        year = ds.forecast_time.dt.year.to_index().unique()
        week = (list(np.arange(1,54)))
        weeks = week * len(year)
        years = np.repeat(year,len(week))
        ds.coords["week"] = ("forecast_time", weeks)
        ds.coords['week'].attrs['description'] = "This week represents the number of forecast_time starting from 1 to 53. Note: This week is different from the ISO week from groupby('forecast_time.weekofyear'), see https://en.wikipedia.org/wiki/ISO_week_date and https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge/-/issues/29"
        ds.coords["year"] = ("forecast_time", years)
        ds.coords['year'].attrs['long_name'] = "calendar year"
    return ds


def make_probabilistic(ds, tercile_edges, member_dim='realization', mask=None, groupby_coord='week'):
    """Compute probabilities from ds (observations or forecasts) based on tercile_edges."""
    # broadcast
    ds = add_year_week_coords(ds)
    tercile_edges = tercile_edges.sel({groupby_coord: ds.coords[groupby_coord]})
    bn = ds < tercile_edges.isel(category_edge=0, drop=True)  # below normal
    n = (ds >= tercile_edges.isel(category_edge=0, drop=True)) & (ds < tercile_edges.isel(category_edge=1, drop=True))  # normal
    an = ds >= tercile_edges.isel(category_edge=1, drop=True)  # above normal

    if member_dim in ds.dims:# using this, the function can deal with nans correctly
        denominator = ds.notnull().sum(member_dim)
        bn = bn.sum(member_dim)/denominator
        an = an.sum(member_dim)/denominator
        n = n.sum(member_dim)/denominator
# =============================================================================
#     if member_dim in ds.dims:
#         bn = bn.mean(member_dim)
#         an = an.mean(member_dim)
#         n = n.mean(member_dim)
# =============================================================================
    ds_p = xr.concat([bn, n, an],'category').assign_coords(category=['below normal', 'near normal', 'above normal'])
    if mask is not None:
        ds_p = ds_p.where(mask)
    if 'tp' in ds_p.data_vars:
        # mask arid grid cells where category_edge are too close to 0
        # we are using a dry mask as in https://doi.org/10.1175/MWR-D-17-0092.1
        tp_arid_mask = tercile_edges.tp.isel(category_edge=0, drop=True) > 0.01#lead_time=0,
        ds_p['tp'] = ds_p['tp'].where(tp_arid_mask)
    ds_p['category'].attrs = {'long_name': 'tercile category probabilities', 'units': '1',
                        'description': 'Probabilities for three tercile categories. All three tercile category probabilities must add up to 1.'}
    if 'tp' in ds_p.data_vars:
        ds_p['tp'].attrs = {'long_name': 'Probability of total precipitation in tercile categories', 'units': '1',
                          'comment': 'All three tercile category probabilities must add up to 1.',
                          'variable_before_categorization': 'https://confluence.ecmwf.int/display/S2S/S2S+Total+Precipitation'
                         }
    if 't2m' in ds_p.data_vars:
        ds_p['t2m'].attrs = {'long_name': 'Probability of 2m temperature in tercile categories', 'units': '1',
                          'comment': 'All three tercile category probabilities must add up to 1.',
                          'variable_before_categorization': 'https://confluence.ecmwf.int/display/S2S/S2S+Surface+Air+Temperature'
                          }
    if 'year' in ds_p.coords:
        del ds_p.coords['year']
    if groupby_coord in ds_p.coords:
        ds_p = ds_p.drop(groupby_coord)
    return ds_p


def skill_by_year(preds, cache_path = '/data/S2S/ecmwf_biweekly_weeks/test-input', adapt=False):
    """Returns pd.Dataframe of RPSS per year."""
    # similar verification_RPSS.ipynb
    # as scorer bot but returns a score for each year
    xr.set_options(keep_attrs=True)
    
    # from root
    #renku storage pull data/forecast-like-observations_2020_biweekly_terciled.nc
    #renku storage pull data/hindcast-like-observations_2000-2019_biweekly_terciled.nc
    #cache_path = '../template/data'
    if 2020 in preds.forecast_time.dt.year:
        obs_p = xr.open_dataset(f'{cache_path}/forecast-like-observations_2020_biweekly_terciled_t2m.nc').sel(forecast_time=preds.forecast_time)
    else:
        obs_p = xr.open_dataset(f'{cache_path}/hindcast-like-observations_2000-2019_biweekly_terciled_t2m.zarr', engine='zarr').sel(forecast_time=preds.forecast_time)
    
    # ML probabilities
    fct_p = preds

    
    # climatology
    clim_p = xr.DataArray([1/3, 1/3, 1/3], dims='category', coords={'category':['below normal', 'near normal', 'above normal']}).to_dataset(name='tp')
    clim_p['t2m'] = clim_p['tp']
    
    if adapt:
        # select only obs_p where fct_p forecasts provided
        for c in ['longitude', 'latitude', 'forecast_time', 'lead_time']:
            obs_p = obs_p.sel({c:fct_p[c]})
        obs_p = obs_p[list(fct_p.data_vars)]
        clim_p = clim_p[list(fct_p.data_vars)]
    
    else:
        # check inputs
        assert_predictions_2020(obs_p)
        assert_predictions_2020(fct_p)
        
    # rps_ML
    rps_ML = xs.rps(obs_p, fct_p, category_edges=None, dim=[], input_distributions='p').compute()
    # rps_clim
    rps_clim = xs.rps(obs_p, clim_p, category_edges=None, dim=[], input_distributions='p').compute()

    ## RPSS
    # penalize # https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge-template/-/issues/7
    expect = obs_p.sum('category')
    expect = expect.where(expect > 0.98).where(expect < 1.02)  # should be True if not all NaN

    # https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge-template/-/issues/50
    rps_ML = rps_ML.where(expect, other=2)  # assign RPS=2 where value was expected but NaN found

    # following Weigel 2007: https://doi.org/10.1175/MWR3280.1
    rpss = 1 - (rps_ML.groupby('forecast_time.year').mean() / rps_clim.groupby('forecast_time.year').mean())
    # clip
    rpss = rpss.clip(-10, 1)
    
    # weighted area mean
    weights = np.cos(np.deg2rad(np.abs(rpss.latitude)))
    # spatially weighted score averaged over lead_times and variables to one single value
    scores = rpss.sel(latitude=slice(None, -60)).weighted(weights).mean('latitude').mean('longitude')
    scores = scores.to_array().mean(['lead_time', 'variable'])
    return scores.to_dataframe('RPSS')


def assert_predictions_2020(preds_test, exclude='weekofyear'):
    """Check the variables, coordinates and dimensions of 2020 predictions."""
    
    # is dataset
    assert isinstance(preds_test, xr.Dataset)

    # has both vars: tp and t2m
    if 'data_vars' in exclude:
        assert 'tp' in preds_test.data_vars
        assert 't2m' in preds_test.data_vars
    
    ## coords
    # ignore weekofyear coord if not dim
    if 'weekofyear' in exclude and 'weekofyear' in preds_test.coords and 'weekofyear' not in preds_test.dims:
        preds_test = preds_test.drop('weekofyear')
    
    # forecast_time
    if 'forecast_time' in exclude:
        d = pd.date_range(start='2020-01-02', freq='7D', periods=53)
        forecast_time = xr.DataArray(d, dims='forecast_time', coords={'forecast_time':d}, name='forecast_time')
        assert_equal(forecast_time,  preds_test['forecast_time'])

    # longitude
    if 'longitude' in exclude:
        lon = np.arange(0., 360., 1.5)
        longitude = xr.DataArray(lon, dims='longitude', coords={'longitude': lon}, name='longitude')
        assert_equal(longitude, preds_test['longitude'])

    # latitude
    if 'latitude' in exclude:
        lat = np.arange(-90., 90.1, 1.5)[::-1]
        latitude = xr.DataArray(lat, dims='latitude', coords={'latitude': lat}, name='latitude')
        assert_equal(latitude, preds_test['latitude'])
    
    # lead_time
    if 'lead_time' in exclude:
        lead = [pd.Timedelta(f'{i} d') for i in [1, 7, 14, 21, 28, 35]]
        lead_time = xr.DataArray(lead, dims='lead_time', coords={'lead_time': lead}, name='lead_time')
        assert_equal(lead_time, preds_test['lead_time'])
    
    # category
    if 'category' in exclude:
        cat = np.array(['below normal', 'near normal', 'above normal'], dtype='<U12')
        category = xr.DataArray(cat, dims='category', coords={'category': cat}, name='category')
        assert_equal(category, preds_test['category'])
    
    # size
    if 'size' in exclude:
        size_in_MB = float(format_bytes(preds_test.nbytes).split(' ')[0])
        # todo: refine for dtypes
        assert size_in_MB > 50
        assert size_in_MB < 250
    
    # no other dims
    if 'dims' in exclude:
        assert set(preds_test.dims) - {'category', 'forecast_time', 'latitude', 'lead_time', 'longitude'} == set()
    


# ============================================================
# DATA LOADING UTILITIES
# ============================================================



def get_data(varlist, path_data, drop_negative_tp='zero'):
    """
    wraps load_data to load hindcasts and observations for a given set of variables.
    also creates a mask for the missing values (grid-cells with at least one missing
    value in 2000-2019 are set to nan.)

    Parameters
    ----------
    var_list: list, list of variables
    path_data: string or 2-element list, if string, it has to be one of {'local', 'server'},
               if 2-element list, it is expected to be the user-defined path to the data
    drop_negative_tp: either True or 'zero', if 'zero' negative precipitation values are set to zero,
                      if True, negative precipitation values are replaced by NA.

    Returns
    -------
    hind: DataSet, contains hindcasts for the specified hindcasts
    obs: DataSet, observations for the hindcast period
    obs_terciled: DataSet, one-hot representation of observations split in terciles, for hindcast period
    mask: DataSet, mask for the missing values (grid-cells with at least one missing value in 2000-2019
                   are set to NA)
    """
    # hindcasts
    hind = get_data_single(data='hind_2000-2019', aggregation='biweekly',
                           path_data=path_data, var_list=varlist, drop_negative_tp=drop_negative_tp)

    # observations corresponding to hindcasts
    obs = load_data(data='obs_2000-2019', aggregation='biweekly', path=path_data)
    # terciled
    obs_terciled = load_data(data='obs_terciled_2000-2019', aggregation='biweekly',
                             path=path_data)

    # mask: same missing values at all forecast_times, only used for label data
    # notnull=True --> set to 1
    mask = xr.where(obs.notnull(), 1, np.nan).mean('forecast_time', skipna=False)

    return hind, obs, obs_terciled, mask


def get_data_single(data, path_data, var_list=['tp', 't2m'],
                    drop_negative_tp='zero', aggregation='biweekly'):
    """
    wraps load_data to load single data files and drops superfluous coordinates and deals with negative
    precipitation amounts.

    Parameters
    ----------
    data: string,  one of {'hind_2000-2019', 'obs_2000-2019', 'obs_terciled_2000-2019',
          'obs_tercile_edges_2000-2019', 'forecast_2020', 'obs_2020', 'obs_terciled_2020'}
    path_data: string or 2-element list, if string, it has to be one of {'local', 'server'},
               if 2-element list, it is expected to be the user-defined path to the data
    var_list: list, list of variables
    drop_negative_tp: either True or 'zero', if 'zero' negative precipitation values are set to zero,
                      if True, negative precipitation values are replaced by NA.
    aggregation: string, one of {'biweekly','weekly'}

    Returns
    -------
    dat: DataSet, with some processing already done
    """
    dat = load_data(data=data, aggregation=aggregation,
                    path=path_data, var_list=var_list)

    # hind and forecasts
    if 'realization' in dat.coords:
        dat = dat[var_list]
        dat = clean_coords(dat)
        dat = clean_data(dat, drop_negative_tp=drop_negative_tp)

    return dat


def load_data(data='hind_2000-2019', aggregation='biweekly', path='server',
              var_list=['tp', 't2m']):
    """
    loads .nc and .zarr files

    Parameters
    ----------
    data: string, one of {'hind_2000-2019', 'obs_2000-2019', 'obs_terciled_2000-2019',
          'obs_tercile_edges_2000-2019', 'forecast_2020', 'obs_2020', 'obs_terciled_2020'}
    aggregation: string, one of {'biweekly','weekly'}
    path: string or 2-element list, if string, it has to be one of {'local', 'server'},
          if 2-element list, it is expected to be the user-defined path to the data
    var_list: list, list of variables

    Returns
    -------
    dat: DataSet, unmodified

    """

    cache_path, path_add_vars, path_model, path_pred, path_results, path_add_vars2 = get_paths(path)

    if data == 'hind_2000-2019':
        dat_list = []
        for var in var_list:
            if (var == 'tp') or (var == 't2m'):
                if aggregation == 'biweekly':
                    dat_item = xr.open_zarr(
                        '{}/ecmwf_hindcast-input_2000-2019_{}_deterministic_{}.zarr'.format(cache_path, aggregation, var),
                        consolidated=True)
                else:
                    dat_item = xr.open_zarr(
                        '{}/ecmwf_hindcast-input_2000-2019_{}_deterministic_{}.zarr'.format(path_add_vars, aggregation, var),
                        consolidated=True)
                var_list = [i for i in var_list if i not in ['tp', 't2m']]
            else:
                dat_item = xr.open_zarr(
                    '{}/ecmwf_hindcast-input_2000-2019_{}_deterministic_{}.zarr'.format(path_add_vars, aggregation,
                                                                                        var), consolidated=True)
                if (var == 'gh200') or (var == 'gh500') or (var == 'gh850') or (var == 'u200') or (var == 'u500') or (var == 'u850')\
                        or (var == 'v200') or (var == 'v500') or (var == 'v850') or (var == 'q200') or (var == 'q500') or (var == 'q850')\
                        or (var == 't200') or (var == 't500') or (var == 't850'):
                    dat_item = dat_item.reset_coords('plev', drop=True)
                if (var == 'trr'):
                    dat_item = dat_item.reset_coords('nominal_top', drop=True)

            dat_list.append(dat_item)
        dat = xr.merge(dat_list)

    elif data == 'obs_2000-2019':
        dat = xr.open_zarr(f'{cache_path}/hindcast-like-observations_2000-2019_biweekly_deterministic_tp.zarr',
                           consolidated=True)

    elif data == 'obs_terciled_2000-2019':
        dat = xr.open_zarr(f'{cache_path}/hindcast-like-observations_2000-2019_biweekly_terciled_tp.zarr',
                           consolidated=True)

    elif data == 'obs_tercile_edges_2000-2019':
        dat = xr.open_dataset(f'{cache_path}/hindcast-like-observations_2000-2019_biweekly_tercile-edges_tp.nc')

    elif data == 'forecast_2020':
        dat_list = []
        for var in var_list:
            if (var == 'tp') or (var == 't2m'):
                if aggregation == 'biweekly':
                    dat_item = xr.open_zarr(
                        '{}/ecmwf_forecast-input_2020_{}_deterministic_{}.zarr'.format(path_add_vars2, aggregation, var),
                        consolidated=True)
                else:
                    dat_item = xr.open_zarr(
                        '{}/ecmwf_forecast-input_2020_{}_deterministic_{}.zarr'.format(path_add_vars2, aggregation, var),
                        consolidated=True)
                var_list = [i for i in var_list if i not in ['tp', 't2m']]  # needed since tp and t2m are loaded jointly
            else:
                dat_item = xr.open_zarr(
                    '{}/ecmwf_forecast-input_2020_{}_deterministic_{}.zarr'.format(path_add_vars2, aggregation, var),
                    consolidated=True)
                if (var == 'gh200') or (var == 'gh500') or (var == 'gh850') or (var == 'u200') or (var == 'u500') or (var == 'u850')\
                        or (var == 'v200') or (var == 'v500') or (var == 'v850') or (var == 'q200') or (var == 'q500') or (var == 'q850')\
                        or (var == 't200') or (var == 't500') or (var == 't850'):
                    dat_item = dat_item.reset_coords('plev', drop=True)
                if (var == 'trr'):
                    dat_item = dat_item.reset_coords('nominal_top', drop=True)
            dat_list.append(dat_item)
        dat = xr.merge(dat_list)

    elif data == 'obs_2020':
        dat = xr.open_zarr(f'{path_add_vars2}/forecast-like-observations_2020_biweekly_deterministic_tp.zarr',
                           consolidated=True)

    elif data == 'obs_terciled_2020':
        dat = xr.open_dataset(f'{path_add_vars2}/forecast-like-observations_2020_biweekly_terciled_tp.nc')

    elif data == 'ecmwf_baseline':
        dat = xr.open_dataset(f'{path_add_vars2}/ecmwf_recalibrated_benchmark_2020_biweekly_terciled_tp.nc')
    else:
        print("specified data name is not valid")

    return dat


def clean_coords(dat):
    """
    remove superfluous coordinates

    Parameters
    -------
    dat: DataSet
    """

    if 'sm20' in dat.keys():
        dat = dat.isel(depth_below_and_layer=0).reset_coords('depth_below_and_layer', drop=True)
    if 'msl' in dat.keys():
        dat = dat.isel(meanSea=0).reset_coords('meanSea', drop=True)
    return dat


def clean_data(dat, drop_negative_tp='zero'):
    """

    Parameters
    ----------
    dat: DataSet
    drop_negative_tp: either True or 'zero', if 'zero' negative precipitation values are set to zero,
                      if True, negative precipitation values are replaced by NA.

    """
    print('drop negative tp values: ', drop_negative_tp)
    # set negative tp values to nan
    if ('tp' in dat.keys()) and (drop_negative_tp == True):
        dat['tp'] = xr.where(dat.tp < 0, np.nan, dat.tp)
    # set to zero
    elif ('tp' in dat.keys()) and (drop_negative_tp == 'zero'):
        dat['tp'] = xr.where(dat.tp < 0, 0, dat.tp)
    return dat


def get_basis(out_field, r_basis):
    """ returns a set of basis functions for the input field, adapted from Scheuerer et al. (2020)

    Parameters
    ----------
    out_field: DataArray, basis functions for these lat lon coordinates will be created
    r_basis: int, radius of support of basis functions, the distance between centers of
             basis functions is half this radius, should be chosen depending on input field size.

    Returns
    -------
    basis: values for basis functions over out_field
    lats: lats of input field
    lons: lons of input field
    n_xy: number of grid points in input field
    n_basis: number of basis functions
    """

    # distance between centers of basis functions
    dist_basis = r_basis / 2
    lats = out_field.latitude
    lons = out_field.longitude

    # number of basis functions
    n_basis = int(np.ceil((lats[0] - lats[-1]) / dist_basis + 1) * np.ceil((lons[-1] - lons[0]) / dist_basis + 1))

    # grid coords
    lon_np = lons
    lat_np = lats

    length_lon = len(lon_np)
    length_lat = len(lat_np)

    lon_np = np.outer(lon_np, np.ones(length_lat)).reshape(int(length_lon * length_lat))
    lat_np = np.outer(lat_np, np.ones(length_lon)).reshape(int(length_lon * length_lat))

    # number of grid points
    n_xy = int(length_lon * length_lat)

    # centers of basis functions
    lon_ctr = np.arange(lons[0], lons[-1] + dist_basis, dist_basis)
    length_lon_ctr = len(lon_ctr)  # number of center points in lon direction

    lat_ctr = np.arange(lats[0], lats[-1] - dist_basis, - dist_basis)
    length_lat_ctr = len(lat_ctr)  # number of center points in lat direction

    lon_ctr = np.outer(lon_ctr, np.ones(length_lat_ctr)).reshape(int(n_basis))
    lat_ctr = np.outer(np.ones(length_lon_ctr), lat_ctr).reshape(int(n_basis))

    # compute distances between fct grid and basis function centers
    dst_lon = np.abs(np.subtract.outer(lon_np, lon_ctr).reshape(len(lons), len(lats), n_basis))  # 10,14
    dst_lon = np.swapaxes(dst_lon, 0, 1)
    dst_lat = np.abs(np.subtract.outer(lat_np, lat_ctr).reshape(len(lats), len(lons), n_basis))  # 14,10

    dst = np.sqrt(dst_lon ** 2 + dst_lat ** 2)
    dst = np.swapaxes(dst, 0, 1).reshape(n_xy, n_basis)

    # define basis functions
    basis = np.where(dst > r_basis, 0., (1. - (dst / r_basis) ** 3) ** 3)  # main step, zero outside,
    basis = basis / np.sum(basis, axis=1)[:, None]  # normalization at each grid point
    # nbs = basis.shape[1]

    return basis, lats, lons, n_xy, n_basis


# ============================================================
# PREPROCESSING UTILITIES
# ============================================================



def preprocess_input(train, v, path_data, lead_time, valid=None, ens_mean=True, spatial_std=True):
    """
    preprocess ensemble forecast input:
        - compute ensemble mean
        - remove annual cycle from non-target features
        - subtract tercile edges from target variable features
        - standardize using standard deviation

    Parameters
    ----------
    train: DataSet, contains ensemble forecasts for one lead time from training data
    v: string, target variable
    path_data: string, path used for load_data
    lead_time: int, lead time in fct (for isel)
    valid: DataSet, if you also want to preprocess validation data
    ens_mean: bool, whether to only use th ensemble mean
    spatial_std: bool, standard deviation used to normalize all features,
                       if False, standard deviation across lat/lon and average over forecast time,
                       if True, std is a spatial field,

    Returns
    -------
    train: DataSet, preprocessed
    valid: DataSet, if given, also preprocessed

    """

    print('use ensemble mean: ', ens_mean)
    if valid != None:
        assert (train.keys() == valid.keys()), f'train and validation set do not contain the same variables: train {list(train.keys())}, val {list(valid.keys())}'

    ls_target_vars = [i for i in train.keys() if i in [v]]  # list of target var features
    ls_rm_annual = [i for i in train.keys() if i not in ['lsm', v]]  # list of vars for which annual cycle should be removed

    # use ensemble mean
    if ens_mean == True:
        train = train.mean('realization')

    if valid != None:
        reference = train.copy(deep=True)

    # remove annual cycle from non-target features
    # don't remove annual cycle from land-sea mask and target variable
    if len(ls_rm_annual) > 0:
        train.update(rm_annualcycle(train[ls_rm_annual], train[ls_rm_annual]))

    # preprocessing for target var: subtract tercile edges from target variable
    if len(ls_target_vars) > 0:
        target_features = target_features_tercile_distance(v, train[ls_target_vars],
                                                           path_data, lead_time)
        # merge all features and drop original target variable feature
        train = xr.merge([train, target_features])
        train = train.drop_vars(ls_target_vars)

    # standardization
    if spatial_std == False:  # standard deviation across lat/lon of the ensemble mean
        # todo: weighted std
        train_std = train.std(('latitude', 'longitude')).mean('forecast_time')
    else:
        train_std = train.std('forecast_time')
    train = train / train_std

    # =============================================================================
    # do the same for valid using train whenever necessary
    # =============================================================================
    if valid != None:

        # use ensemble mean
        if ens_mean == True:
            valid = valid.mean('realization')

        # remove annual cycle from non-target features
        valid.update(rm_annualcycle(valid[ls_rm_annual], reference[ls_rm_annual]))

        # preprocessing for target var: subtract tercile edges from target variable
        if len(ls_target_vars) > 0:
            target_features_valid = target_features_tercile_distance(v, valid[ls_target_vars],
                                                                     path_data, lead_time)
            # merge all features and drop original target variable feature
            valid = xr.merge([valid, target_features_valid])
            valid = valid.drop_vars(ls_target_vars)

        # standardization
        valid = valid / train_std

    if valid != None:
        return train, valid
    else:
        return train


def rm_annualcycle(ds, ds_train):
    """
    remove annual cycle for each location

    Parameters
    ----------
    ds: DataSet, this is the dataset from which the annual cycle should be removed
    ds_train: DataSet, this is the dataset from which the annual cycle is computed

    Returns
    -------
    ds_stand: ds with removed annual cycle
    """

    ds = add_year_week_coords(ds)
    ds_train = add_year_week_coords(ds_train)

    if 'realization' in ds_train.coords:  # always use train data to compute the annual cycle
        ann_cycle = ds_train.mean('realization').groupby('week').mean(['forecast_time'])
    else:
        ann_cycle = ds_train.groupby('week').mean(['forecast_time'])

    ds_stand = ds - ann_cycle

    ds_stand = ds_stand.sel({'week': ds.coords['week']})
    ds_stand = ds_stand.drop(['week', 'year'])

    return ds_stand


def target_features_tercile_distance(v, fct, path_data, lead_time):
    """
    Compute distance from ensemble mean to the tercile edges,
    wrapper to rm_tercile_edges that saves distances under new
    variable names lower_{v}, upper_{v}

    Parameters
    ----------
    v: string, target variable
    fct: DataSet, only contains target variable v
    path_data: string, path used for load-data to load the tercile edges
    lead_time: int, lead time in fct (for isel)

    Returns
    -------
    dist_renamed: DataSet
    """
    tercile_edges = load_data(data='obs_tercile_edges_2000-2019',
                              aggregation='biweekly', path=path_data
                              ).isel(lead_time=lead_time)[v]
    dist = rm_tercile_edges(fct, tercile_edges)

    # rename the new tercile features
    ls_dist = []
    for var in list(dist.keys()):
        ls_dist.append(
            dist[var].assign_coords(category_edge=['lower_{}'.format(var),
                                                   'upper_{}'.format(var)]
                                    ).to_dataset(dim='category_edge')
        )
    dist_renamed = xr.merge(ls_dist)

    return dist_renamed


def rm_tercile_edges(ds, tercile_edges):
    """
    compute distance to tercile edges

    Parameters
    ----------
    ds: DataSet
    tercile_edges: DataSet, loaded tercile edges

    Returns
    -------
    ds_stand: distance to tercile edges
    """

    ds = add_year_week_coords(ds)

    ds_stand = tercile_edges - ds

    ds_stand = ds_stand.sel({'week': ds.coords['week']})
    ds_stand = ds_stand.drop(['week', 'year'])

    return ds_stand


# ============================================================
# DATA GENERATORS
# ============================================================


class DataGeneratorGlobal(keras.utils.Sequence):
    """
    data generator used for global training
    input_lats=128, input_lons=256,
    """

    def __init__(self, fct, verif, region='global',
                 batch_size=32, shuffle=True, load=False):
        """
        Parameters
        ----------
        fct: DataSet, global preprocessed fields, coordinates: latitude, longitude, forecast_time
        verif: DataSet, global preprocessed fields, coordinates: latitude, longitude, forecast_time
        region: string, domain the model should be trained on, either global or europe
        batch_size: int
        shuffle: bool, if True, data is shuffled
        load: bool, if True, data is loaded into RAM.

        Returns
        -------
        data generator object
        """

        self.batch_size = batch_size
        self.shuffle = shuffle

        if region == 'europe':
            fct = flip_antimeridian(fct, to='Pacific', lonn='longitude')
            verif = flip_antimeridian(verif, to='Pacific', lonn='longitude')

            # European domain that is divisible by 8 in lat lon direction
            fct = fct.sel(latitude=slice(90, 7), longitude=slice(-71, 73))
            verif = verif.sel(latitude=slice(79, 19), longitude=slice(-59, 60))
        else:  # global
            fct = pad_earth(fct, pad_args=8)
            fct = fct.isel(latitude=slice(4, -5))

        # create self. ... data
        self.fct_data = fct.transpose('forecast_time', ...)
        self.verif_data = verif.transpose('forecast_time', ...)

        self.n_samples = self.fct_data.forecast_time.shape[0]

        self.on_epoch_end()

        if load: self.fct_data.load()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, i):
        """Generate one batch of data"""

        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]

        X_x_list = []
        y_list = []
        for j in idxs:
            X_x_list.append(np.expand_dims(self.fct_data.isel(forecast_time=j
                                                            ).fillna(0.).to_array(
                            ).transpose(..., 'longitude', 'latitude', 'variable'
                                        ).values, axis=0))

            y_list.append(np.expand_dims(self.verif_data.isel(forecast_time=j
                                                              ).fillna(0.
                                         ).transpose(..., 'longitude', 'latitude', 'category'
                                                     ).values, axis=0))

        X_x = np.concatenate(X_x_list)
        X = [X_x]
        y = np.concatenate(y_list)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""

        self.idxs = np.arange(self.n_samples)

        if self.shuffle == True:
            np.random.shuffle(self.idxs)  # in place
            print('reshuffled idxs')


class DataGeneratorMultipatch(keras.utils.Sequence):
    """ data generator used for patch-wise training"""

    def __init__(self, fct, verif, model, input_dims, output_dims, mask_v, region=None, batch_size=32, shuffle=True,
                 load=False, reduce_sample_size=None, patch_stride=2, used_members=11, fraction=0 / 8, weighted=False):
        """
        data generator for using data from all valid patches of the selected region

        Parameters
        ----------
        model
        fct: DataSet, global preprocessed fields, coordinates: latitude, longitude, forecast_time
        verif: DataSet, global preprocessed fields, coordinates: latitude, longitude, forecast_time
        input_dims: int, size of input domain
        output_dims: int, size of output domain
        mask_v: DataArray, no lead time coordinates, target variable already selected
        region: string, if None, all valid  patches are used, ow only valid patches from the
                respective region are used (poles, midlats, subtropics, tropics)
        batch_size: int
        shuffle: bool, if True, data is shuffled
        load: bool, if True, data is loaded into RAM.
        reduce_sample_size: int or None, if not None, provide integer that indicates
                            the fraction of used patches from the list of valid patches
        used_members: int, only used if fct has realization coordinates,
                      specifies how many ensemble members should be used
        fraction: float, fraction of grid cell that can be NA for a valid patch
        weighted: bool, if True weights for each grid cell are returned for every sample

        Returns
        -------
        data generator object
        """

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.model = model
        if model.model_architecture == 'basis_func':
            self.basis = model.basis
            self.clim_probs = model.clim_probs

        self.input_dims = input_dims
        self.output_dims = output_dims

        self.weighted = weighted

        if 'realization' in fct:
            # select members that should be used
            # do this as early as possible, since padding is very memory-inefficient
            if used_members > 11:
                print('invalid value in used_members, set to 11')
                use_members = 11
            fct = fct.isel(realization=slice(0, used_members))

        # pad
        pad_args = [input_dims, output_dims]
        fct = pad_earth(fct, pad_args)
        verif = pad_earth(verif, pad_args)
        # mask gets padded in get_all_valid_patches

        # create self. ... data
        self.fct_data = fct.transpose('forecast_time', ...)
        self.verif_data = verif.transpose('forecast_time', ...)

        # create reference df with coordinates of all valid patches
        # patch is valid if it contains less NA than indicated by 'fraction'
        valid_coords_raw = get_all_valid_patches(mask_v, output_dims, pad_args=pad_args, fraction=fraction,
                                                 patch_stride=patch_stride, region=region)

        if reduce_sample_size != None:
            valid_coords_raw = valid_coords_raw.iloc[
                               0:int(valid_coords_raw.shape[0] / reduce_sample_size)]  # e.g. 10 or 5

        # expand valid_coords_raw by forecast_time dimension
        n_samples_time = self.fct_data.forecast_time.size
        self.valid_coords = pd.concat([valid_coords_raw] * n_samples_time, ignore_index=True)  # 2'976'480 rows
        new_col = np.repeat(np.arange(n_samples_time), valid_coords_raw.shape[0])
        self.valid_coords['time'] = new_col

        if 'realization' in self.fct_data:
            valid_coords = self.valid_coords.copy()
            n_samples_realization = self.fct_data.realization.size
            self.valid_coords = pd.concat([valid_coords] * n_samples_realization, ignore_index=True)
            new_col = np.repeat(np.arange(n_samples_realization), valid_coords.shape[0])
            self.valid_coords['realization'] = new_col

        self.n_samples = self.valid_coords.shape[0]

        self.on_epoch_end()

        if self.weighted == True:
            weights = np.cos(np.deg2rad(np.abs(mask_v.latitude)))
            self.mask_weighted = (mask_v.transpose('longitude', 'latitude').fillna(0) * weights)
            self.mask_weighted = pad_earth(self.mask_weighted, pad_args)
            self.mask_weighted.load()

        if load: self.fct_data.load()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, i):
        """Generate one batch of data"""
        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]

        X_x_list = []
        y_list = []
        weight_list = []

        for j in idxs:

            time = self.valid_coords['time'].iloc[j]
            lat = self.valid_coords['latitude'].iloc[j]
            lon = self.valid_coords['longitude'].iloc[j]

            if 'realization' in self.fct_data:
                realization = self.valid_coords['realization'].iloc[j]
                X_x_list.append(np.expand_dims(self.fct_data.isel(forecast_time=time,
                                                                  latitude=self.get_patch_slices(lat, lon, 'input_lats',
                                                                                                 self.input_dims,
                                                                                                 self.output_dims),
                                                                  longitude=self.get_patch_slices(lat, lon,
                                                                                                  'input_lons',
                                                                                                  self.input_dims,
                                                                                                  self.output_dims),
                                                                  realization=realization
                                                                  ).fillna(0.).to_array().transpose(..., 'longitude',
                                                                                                    'latitude',
                                                                                                    'variable').values,
                                               axis=0))
            else:
                X_x_list.append(np.expand_dims(self.fct_data.isel(forecast_time=time,
                                                                  latitude=self.get_patch_slices(lat, lon, 'input_lats',
                                                                                                 self.input_dims,
                                                                                                 self.output_dims),
                                                                  longitude=self.get_patch_slices(lat, lon,
                                                                                                  'input_lons',
                                                                                                  self.input_dims,
                                                                                                  self.output_dims)
                                                                  ).fillna(0.).to_array().transpose(..., 'longitude',
                                                                                                    'latitude',
                                                                                                    'variable').values,
                                               axis=0))

            if self.weighted:
                weight_list.append(np.expand_dims(self.mask_weighted.isel(
                    latitude=self.get_patch_slices(lat, lon, 'output_lats', self.input_dims, self.output_dims),
                    longitude=self.get_patch_slices(lat, lon, 'output_lons', self.input_dims, self.output_dims)
                    ).transpose('longitude', 'latitude').values, axis=0))

            y_list.append(np.expand_dims(self.verif_data.isel(forecast_time=time,
                                                              latitude=self.get_patch_slices(lat, lon, 'output_lats',
                                                                                             self.input_dims,
                                                                                             self.output_dims),
                                                              longitude=self.get_patch_slices(lat, lon, 'output_lons',
                                                                                              self.input_dims,
                                                                                              self.output_dims)
                                                              ).fillna(0.).transpose(..., 'longitude', 'latitude',
                                                                                     'category').values,
                                         axis=0))

        X_x = np.concatenate(X_x_list)
        if self.model.model_architecture == 'basis_func':
            X_basis = np.repeat(self.basis[np.newaxis, :, :], len(idxs), axis=0)
            X_clim = np.repeat(self.clim_probs[np.newaxis, :, :], len(idxs), axis=0)
            X = [X_x, X_basis, X_clim]
        else:
            X = [X_x]
        y = np.concatenate(y_list)

        if self.weighted:
            weights = np.concatenate(weight_list)
            X = X + [weights, y]

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""

        self.idxs = np.arange(self.n_samples)

        if self.shuffle == True:
            np.random.shuffle(self.idxs)  # in place
            print('reshuffled idxs')

    def get_patch_slices(self, lat, lon, kind, input_dims, output_dims):
        """
        yield slices for latitude and longitude coordinates based on the coordinates
        of a patch corner
        """
        if kind == 'input_lats':
            input_lats = slice(lat - output_dims - int((input_dims - output_dims) / 2) + 1,
                               lat + int((input_dims - output_dims) / 2) + 1)
            return input_lats
        elif kind == 'input_lons':
            input_lons = slice(lon - output_dims - int((input_dims - output_dims) / 2) + 1,
                               lon + int((input_dims - output_dims) / 2) + 1)
            return input_lons
        elif kind == 'output_lats':
            output_lats = slice(lat - output_dims + 1, lat + 1)
            return output_lats
        elif kind == 'output_lons':
            output_lons = slice(lon - output_dims + 1, lon + 1)
            return output_lons
        else:
            print('no valid "kind" argument provided')


def flip_antimeridian(ds, to='Pacific', lonn='lon'):
    """
    # https://git.iac.ethz.ch/utility_functions/utility_functions_python/-/blob/regionmask/xarray.py

    Flip the antimeridian (i.e. longitude discontinuity) between Europe
    (i.e., [0, 360)) and the Pacific (i.e., [-180, 180)).
    Parameters:
    - ds (xarray.Dataset or .DataArray): Has to contain a single longitude
      dimension.
    - to='Pacific' (str, optional): Flip antimeridian to one of
      * 'Europe': Longitude will be in [0, 360)
      * 'Pacific': Longitude will be in [-180, 180)
    - lonn=None (str, optional): Name of the longitude dimension. If None it
      will be inferred by the CF convention standard longitude unit.
    Returns:
    same type as input ds
    """

    attrs = ds[lonn].attrs

    if to.lower() == 'europe' and not antimeridian_pacific(ds, lonn):
        return ds  # already correct, do nothing
    elif to.lower() == 'pacific' and antimeridian_pacific(ds, lonn):
        return ds  # already correct, do nothing
    elif to.lower() == 'europe':
        ds = ds.assign_coords(**{lonn: (ds[lonn] % 360)})
    elif to.lower() == 'pacific':
        ds = ds.assign_coords(**{lonn: (((ds[lonn] + 180) % 360) - 180)})
    else:
        errmsg = 'to has to be one of [Europe | Pacific] not {}'.format(to)
        raise ValueError(errmsg)

    was_da = isinstance(ds, xr.core.dataarray.DataArray)
    if was_da:
        da_varn = ds.name
        if da_varn is None:
            da_varn = 'data'
        enc = ds.encoding
        ds = ds.to_dataset(name=da_varn)

    idx = np.argmin(ds[lonn].data)
    varns = [varn for varn in ds.variables if lonn in ds[varn].dims]
    for varn in varns:
        if xr.__version__ > '0.10.8':
            ds[varn] = ds[varn].roll(**{lonn: -idx}, roll_coords=False)
        else:
            ds[varn] = ds[varn].roll(**{lonn: -idx})

    ds[lonn].attrs = attrs
    if was_da:
        da = ds[da_varn]
        da.encoding = enc
        return da
    return ds


def antimeridian_pacific(ds, lonn=None):
    """Returns True if the antimeridian is in the Pacific (i.e. longitude runs
    from -180 to 180."""
    if lonn is None:
        lonn = get_longitude_name(ds)
    if ds[lonn].min() < 0 or ds[lonn].max() < 180:
        return True
    return False


def get_longitude_name(ds):
    """Get the name of the longitude dimension by CF unit"""
    lonn = []
    if isinstance(ds, xr.core.dataarray.DataArray):
        ds = ds.to_dataset()
    for dimn in ds.dims.keys():
        if 'units' in ds[dimn].attrs and ds[dimn].attrs['units'] in ['degree_east', 'degrees_east']:
            lonn.append(dimn)
    if len(lonn) == 1:
        return lonn[0]
    elif len(lonn) > 1:
        errmsg = 'More than one longitude coordinate found by unit.'
    else:
        errmsg = 'Longitude could not be identified by unit.'
    raise ValueError(errmsg)


def pad_earth(fct, pad_args):
    """
    pad global input field on all sides
    data resolution is 1.5°

    Parameters:
    ----------
    fct: DataSet
    pad_args: int or list, if int, it is the pad, if list, should be [input_dims, output_dims],
              will be used to compute the pad
    """

    if type(pad_args) == list:
        input_dims = pad_args[0]
        output_dims = pad_args[1]
        pad = int((input_dims - output_dims) / 2) + output_dims
    else:
        pad = pad_args

    # create padding for north pole and south pole
    fct_shift = fct.pad(pad_width={'longitude': (0, 120)}, mode='wrap'
                        ).shift({'longitude': 120}
                                ).isel(longitude=slice(120, 120 + int(360 / 1.5)))  # 240/2 = 120

    fct_shift_pad = fct_shift.pad(pad_width={'latitude': (pad, pad)}, mode='reflect')
    shift_pad_south = fct_shift_pad.isel(latitude=slice(len(fct_shift_pad.latitude) - pad,
                                                        len(fct_shift_pad.latitude)))

    shift_pad_north = fct_shift_pad.isel(latitude=slice(0, pad))

    # add pole padding to ftc_train
    fct_lat_pad = xr.concat([shift_pad_north, fct, shift_pad_south], dim='latitude')

    # pad in east-west direction
    fct_padded = fct_lat_pad.pad(pad_width={'longitude': (pad, pad)}, mode='wrap')
    return fct_padded


def get_all_valid_patches(mask, output_dims, pad_args, fraction=1 / 8, patch_stride=2, region=None):
    """
    Parameters
    ----------
    mask: DataArray no lead time coords, targed variable already selected
    output_dims: int, size of output domain
    fraction: float, fraction of grid cell that can be NA for a valid patch.
    pad_args: int, with how many grid cells global field should be padded
    region: string, if None, all valid  patches are used, ow only valid
                    patches from the respective region are used (poles, midlats, subtropics, tropics)
    Returns
    -------
    possible_coords: dataframe with isel coords of valid patches for padded dataarray
                     coords corresponds to bottom right corner of the valid patch.
    """

    if type(pad_args) == list:
        input_dims = pad_args[0]
        output_dims = pad_args[1]
        pad = int((input_dims - output_dims) / 2) + output_dims
    else:
        pad = pad_args

    print('patch stride', patch_stride)
    mask_pad = pad_earth(mask, pad_args)  # make sure that all pixel in the final domain are completely rolled

    mask_pad_new_coords = mask_pad.assign_coords(longitude=np.arange(0, len(mask_pad.longitude)),
                                                 latitude=np.arange(0, len(mask_pad.latitude)))
    rolling = mask_pad_new_coords.rolling(latitude=output_dims, longitude=output_dims, min_periods=1)
    use_patch_out = rolling.construct(latitude='latitude_win',
                                      longitude='longitude_win',
                                      stride=patch_stride).sum(('latitude_win',
                                                                'longitude_win')).compute()
    # the first element of the rolling is the sum of the first element in mask_pad_new_coords and win-1 na values.
    # so, if 16 elements are padded, then 16 rolling elements are created for this region.

    keep_final_new = xr.where(use_patch_out > output_dims ** 2 * (1 - fraction) - 1, 1, 0
                              ).compute()  # -1 since now 1/8 missing is fine

    coords = keep_final_new.to_dataframe().reset_index()
    possible_coords = coords.loc[coords[coords.columns[3]] == 1]

    # ensure that output is only in domain + output_dims -1
    possible_coords = possible_coords.loc[(possible_coords['latitude'] > pad - 1) &
                                          (possible_coords['latitude'] < pad + 121)]
    possible_coords = possible_coords.loc[(possible_coords['longitude'] > pad - 1) &
                                          (possible_coords['longitude'] < pad + 240)]

    if region == 'poles':  # >60
        possible_coords = possible_coords.loc[
            (possible_coords.latitude < pad + 20) | (possible_coords.latitude > pad + 100)]
    elif region == 'midlats':  # >40 <=60
        possible_coords = possible_coords.loc[
            ((possible_coords.latitude >= pad + 20) & (possible_coords.latitude < pad + 34)) |
            ((possible_coords.latitude <= pad + 100) & (possible_coords.latitude > pad + 86))]
    elif region == 'subtropics':  # >23.5 <=40
        possible_coords = possible_coords.loc[
            ((possible_coords.latitude >= pad + 34) & (possible_coords.latitude < pad + 45)) |
            ((possible_coords.latitude <= pad + 86) & (possible_coords.latitude > pad + 75))]
    elif region == 'tropics':  # <=23.5
        possible_coords = possible_coords.loc[
            (possible_coords.latitude >= pad + 45) & (possible_coords.latitude <= pad + 75)]

    return possible_coords


# ============================================================
# TRAINING HELPERS
# ============================================================


def fit_model(self, cnn, dg_train, dg_valid, call_back, delayed_early_stop, cprofile=False):

    # customize optimizer
    if self.optimizer_str == 'adam':  # learn_rate = 0.001
        optimizer = keras.optimizers.Adam(self.learn_rate, decay=self.decay_rate)
        print('adam', self.learn_rate, self.decay_rate)
    elif self.optimizer_str == 'SGD':  # learn_rate = 0.1
        optimizer = keras.optimizers.SGD(learning_rate=self.learn_rate, decay=self.decay_rate)
        print('SGD', self.learn_rate, self.decay_rate)
    else:
        optimizer = keras.optimizers.Adam(self.learn_rate)

    # delayed early stopping
    if delayed_early_stop == True:
        print('delayed early stopping')
        callback = CustomStopper(monitor='val_loss', patience=self.patience,
                                 restore_best_weights=True, min_delta=0,
                                 start_epoch=self.start_epoch)
    else:
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience,
                                                    restore_best_weights=True, min_delta=0.001)

    # compile models
    if self.weighted_loss == True:
        compute_weighted_loss = get_weighted_loss_function(self.train_patches)
        if self.train_patches == True:
            cnn.add_loss(compute_weighted_loss(cnn.target, cnn.out, cnn.weight_mask))
            cnn.compile(loss=None,  # weighted_loss(mask = mask_weighted),
                        metrics=['accuracy', keras.metrics.CategoricalCrossentropy()],
                        optimizer=optimizer)
        else:
            cnn.compile(loss=compute_weighted_loss(mask=self.weights),
                        metrics=['accuracy', keras.metrics.CategoricalCrossentropy()],
                        optimizer=optimizer)
    else:
        cnn.compile(loss=keras.losses.CategoricalCrossentropy(),
                    metrics=['accuracy'], optimizer=optimizer)

    # fit models
    if call_back:
        if cprofile:
            # Create a TensorBoard callback
            logs = "/results/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
                                                             histogram_freq=1,
                                                             profile_batch='5,10',
                                                             write_images=True)
            self.hist = cnn.fit(dg_train, epochs=self.ep, validation_data=dg_valid, callbacks=[callback, tboard_callback])
        else:
            self.hist = cnn.fit(dg_train, epochs=self.ep, validation_data=dg_valid, callbacks=[callback, PrintLearningRate()])

    else:
        self.hist = cnn.fit(dg_train, epochs=self.ep, validation_data=dg_valid,
                       callbacks=[PrintLearningRate()])


def get_weighted_loss_function(train_patches):
    if train_patches == True:
        def compute_weighted_loss(y_true, y_pred, mask):
            cce_func = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
            cce = cce_func(y_true, y_pred)
            cce_weighted = tf.cast(cce, dtype=tf.float32) * mask
            loss_ = tf.math.reduce_sum(cce_weighted) / tf.math.reduce_sum(mask)
            return loss_

    else:
        def compute_weighted_loss(mask):
            # double function needed, since loss can only take y_true and y_pred
            # as input, but we also need to pass the weights
            def loss_(y_true, y_pred):
                cce_func = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
                cce = cce_func(y_true, y_pred)

                mask_tiled = tf.tile(tf.expand_dims(tf.constant(mask, dtype=tf.float32),
                                                    axis=0),
                                     (tf.shape(y_pred)[0], 1, 1))
                cce_weighted = tf.cast(cce, dtype=tf.float32) * mask_tiled
                return tf.math.reduce_sum(cce_weighted) / tf.math.reduce_sum(mask_tiled)

            return loss_

    return compute_weighted_loss


class CustomStopper(keras.callbacks.EarlyStopping):
    def __init__(self, monitor='val_loss', min_delta=0, patience=0,
                 verbose=0, mode='auto', baseline=None, restore_best_weights=False,
                 start_epoch=5):  # add argument for starting epoch
        super(CustomStopper, self).__init__()

        # hand over values from the arguments
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.start_epoch = start_epoch
        #self.best_weights = None  # Initialize best_weights attribute

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch:  # since epoch count starts from 0.
            # print("lagged early stop")
            super().on_epoch_end(epoch, logs)


# print learning rate after every epoch to monitor how the learning rate decay works
class PrintLearningRate(Callback):
    def __init__(self):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        lr = K.eval(self.model.optimizer._decayed_lr(tf.float64))
        print("\nLearning rate at epoch {} is {}".format(epoch, lr))


def save_model_info(model, model_name, v, lead_time, dg_train_len, fold_no, fct_train_keys, time, folder=''):

    if model.call_back == True:
        if model.delayed_early_stop == False:
            early_stop = 'normal'
        else:
            early_stop = 'delayed'
    else:
        early_stop = False
    
    setup = pd.DataFrame(
                 columns=['model_name', 'target_variable', 'lead_time', 'model_architecture',
                          'train_patches', 'weighted_loss', 'fold_no', 'train_time', 'epochs', 'batch_size',
                          'n_trainbatches', 'callback', 'features',
                           'output_dims', 'input_dims', 'patch_stride', 'patch_na', 'region',
                          'optimizer', 'learn_rate', 'learn_decay', 'early_stopping',
                          'filters', 'hidden_nodes', 'n_blocks', 'dropout_rate', 'batch_norm',
                          'radius_basis_func'])

    # params saved as class params
    d = model.__dict__ 
    
    # some cols got new names, so dataframe column names do not match the keys
    d1 = {'radius_basis_func': 'basis_rad', 'batch_size': 'bs', 'epochs':'ep',
          'batch_norm': 'bn', 'optimizer': 'optimizer_str',
          'learn_decay': 'decay_rate'}
    
    # some additional info that needs to be saved
    add_params = {'model_name': model_name, 'target_variable': v, 
                  'lead_time': lead_time, 'n_trainbatches': dg_train_len,
                  'callback': len(model.hist.history.get('accuracy')) - model.patience,
                  'fold_no': fold_no, 'early_stopping': early_stop,
                  'features': fct_train_keys, 'train_time': time}
    
    for k in setup.columns:
        if k in d.keys():
            setup.at[0, k] = d[k]
        elif k in d1.keys():
            if d1[k] in d.keys():
                setup.at[0, k] = d[d1[k]]
        elif k in add_params.keys():
            setup.at[0, k] = add_params[k]
    setup = setup.fillna('-')  
        
    results_ = pd.DataFrame({'accuracy': [model.hist.history.get('accuracy')],
                             'val_accuracy': [model.hist.history.get('val_accuracy')],
                             'loss': [model.hist.history.get('loss')],
                             'val_loss': [model.hist.history.get('val_loss')]})

    results = pd.concat([setup, results_], axis=1)

    if os.path.isfile(f'{folder}/results.csv'):
        results.to_csv(f'{folder}/results.csv', sep=';', index=False, mode='a', header=None)
    else:
        results.to_csv(f'{folder}/results.csv', sep=';', index=False, mode='a')


# ============================================================
# EXPORTED SYMBOLS
# ============================================================
__all__ = [
    'get_data', 'get_basis', 'preprocess_input',
    'DataGeneratorGlobal', 'DataGeneratorMultipatch',
    'fit_model', 'save_model_info', 'get_paths'
]
