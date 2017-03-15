#!/usr/bin/env python

# This started as the notebook errors_and_one_shots_ms_disabled_for_trending.ipynb in the aca_status_flags repo

import os
import logging
import argparse
import numpy as np
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astropy.table import Table, vstack
from Ska.Matplotlib import plot_cxctime, cxctime2plotdate
from Chandra.Time import DateTime
from kadi import events
from Ska.engarchive import fetch
import maude

def get_options():
    parser = argparse.ArgumentParser(
        description="Update plots tracking errors on MSF observations")
    parser.add_argument("--outdir",
                        default='/proj/sot/ska/www/ASPECT/attitude_error_mon/')
    parser.add_argument("--datadir",
                        default='/proj/sot/ska/data/attitude_error_mon/')
    opt = parser.parse_args()
    return opt


def get_obs_table(start, stop, msf):
    """
    Make a data table of obsids with MS filter status, one shot magnitudes, and att errors
    Only include observations that match the supplied MS filter status

    :param start: start time for dwell range to fetch
    :param stop: stop time for range
    :param msf: multiple star filter status ('ENAB' or 'DISA')
    :returns: astropy table of observation data
    """
    m_filter_telem = maude.get_msids('AOACIMSS', start, stop)
    aoacimss = m_filter_telem['data'][0]
    dwells = events.dwells.filter(start=start, stop=stop)
    obs_data = []
    for d in dwells:
        obs = {}
        obsid = d.get_obsid()
        if obsid is None:
            continue
        if obsid == 0:
            continue
        #print obsid, d.start
        # for data earlier than the existence of the AOACIMSS msid,
        # the value can be assumed to be 'ENAB'
        if d.manvr.kalman_start < '2015:252':
            obs['gui_ms'] = 'ENAB'
        elif len(aoacimss['times']) and (d.tstop > aoacimss['times'][-1]):
            continue
        else:
            mid_kalman = ((DateTime(d.manvr.kalman_start).secs
                            + DateTime(d.stop).secs) / 2)
            kal_idx = (np.searchsorted(aoacimss['times'],
                                        mid_kalman))
            obs['gui_ms'] = aoacimss['values'][kal_idx]

        # if the dwell isn't for the type of data we're looking for, just continue
        if obs['gui_ms'] != msf:
            continue

        # Get one shot and maneuver to for *next* obsid
        n = d.get_next()
        if not n:
            continue
        obs['next_obsid'] = n.get_obsid()

        obs['date'] = d.manvr.kalman_start
        obs['datestop'] = d.stop
        obs['time'] = DateTime(d.manvr.kalman_start).secs
        obs['timestop'] = d.tstop

        obs['manvr_angle'] = n.manvr.angle
        obs['one_shot'] = n.manvr.one_shot
        obs['obsid'] = obsid

        all_err = {}
        for err_name, err_msid in zip(['roll_err', 'pitch_err', 'yaw_err'],
                                        ['AOATTER1', 'AOATTER2', 'AOATTER3']):
            err = fetch.Msid(err_msid, d.tstart + 500, d.stop)
            events.dumps.interval_pad = (0, 300)
            err.remove_intervals(events.dumps)
            events.tsc_moves.interval_pad = (0, 300)
            err.remove_intervals(events.tsc_moves)
            if len(err.times):
                err.remove_intervals(events.ltt_bads)
            all_err[err_name] = err

        if not len(all_err['roll_err'].vals):
            continue

        obs['roll_err'] = np.degrees(np.percentile(np.abs(all_err['roll_err'].vals), 99)) * 3600
        point_err = np.sqrt((all_err['pitch_err'].vals ** 2) + (all_err['yaw_err'].vals ** 2))
        obs['point_err'] = np.degrees(np.percentile(point_err, 99)) * 3600
        obs_data.append(obs)

    t = Table(obs_data)['obsid', 'next_obsid', 'time', 'timestop', 'date', 'datestop',
                        'gui_ms', 'manvr_angle', 'one_shot',
                        'roll_err', 'point_err']
    return t

def one_shot_plot(ref_data, msd_data, label=None, min_dwell_time=1000, outdir='.'):
    fig = plt.figure(figsize=(5, 3.5))
    ref_data = ref_data[(ref_data['timestop'] - ref_data['time']) >= min_dwell_time]
    msd_data = msd_data[(msd_data['timestop'] - msd_data['time']) >= min_dwell_time]
    plt.plot(ref_data['manvr_angle'], ref_data['one_shot'], 'b+', markersize=5, markeredgewidth=.8,
             alpha=.4,
             label='last dwell MSF ENAB')
    plt.plot(msd_data['manvr_angle'], msd_data['one_shot'], 'rx', markersize=5, markeredgewidth=.8,
             label='last dwell MSF DISA')
    plt.grid()
    plt.xlim(0, 185)
    plt.ylim(ymin=0)
    plt.ylabel('One Shot (arcsec)')
    plt.xlabel('Manvr Angle (deg)')
    leg = plt.legend(loc='upper left', fontsize=8, numpoints=1, handletextpad=0)
    # make legend semi-transparent
    leg.get_frame().set_alpha(0.5)
    plt.title('One Shot Magnitude ({})'.format(label), fontsize=12, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'one_shot_vs_angle.png'))


def att_err_time_plots(ref_data, msd_data, min_dwell_time=1000, outdir='.'):
    ref_data = ref_data[(ref_data['timestop'] - ref_data['time']) >= min_dwell_time]
    msd_data = msd_data[(msd_data['timestop'] - msd_data['time']) >= min_dwell_time]
    for i, ax in enumerate(['roll', 'point'], 1):
        default_ylims = [0, 1.4]
        if ax == 'roll':
            default_ylims = [0, 12]
        fig = plt.figure(figsize=(5, 3.5))
        ax1 = fig.add_axes([.15, .55, .8, .37])
        plot_cxctime(ref_data['time'], ref_data['{}_err'.format(ax)], 'b+', markersize=5,
                     alpha=.5,
                     label='MSF ENAB')
        plt.ylabel('MSF Enabled\n{} err (arcsec)'.format(ax))
        plt.grid()
        plt.margins(x=.1, y=.25)
        ax2 = fig.add_axes([.15, .1, .8, .37])
        plot_cxctime(msd_data['time'], msd_data['{}_err'.format(ax)], 'rx', markersize=5,
                     label='MSF DISA')
        plt.suptitle('99th percentile {} error magnitude (per obs)'.format(ax, i),
              fontsize=12)
        plt.grid()
        plt.ylabel('MSF Disabled\n{} err (arcsec)'.format(ax))
        plt.margins(x=.1, y=.25)
        ylims = plt.ylim()
        setlims = plt.ylim(np.min([ylims[0], default_ylims[0]]), np.max([ylims[1], default_ylims[1]]))
        ax1.set_ylim(setlims)
        plt.setp(ax1.get_xticklabels(), visible=True)
        plt.setp(ax1.get_xticklabels(), fontsize=7)
        plt.setp(ax2.get_xticklabels(), fontsize=7)
        plt.setp(ax1.get_yticklabels(), fontsize=7)
        plt.setp(ax2.get_yticklabels(), fontsize=7)
        plt.setp(ax1.get_xticklabels(), rotation=0)
        plt.setp(ax1.get_xticklabels(), horizontalalignment='center')
        plt.savefig(os.path.join(outdir, '{}_err_vs_time.png'.format(ax)))


def att_err_hist(ref_data, msd_data, label=None, min_dwell_time=1000, outdir='.'):
    ref_data = ref_data[(ref_data['timestop'] - ref_data['time']) >= min_dwell_time]
    msd_data = msd_data[(msd_data['timestop'] - msd_data['time']) >= min_dwell_time]
    for i, ax in enumerate(['roll', 'point'], 1):
        fig = plt.figure(figsize=(5, 3.5))
        bin_width = .05
        lim = np.max([1.4, np.max(msd_data['point_err'])])
        if ax == 'roll':
            bin_width = .25
            lim = np.max([12, np.max(msd_data['roll_err'])])
        bins = np.arange(0, lim + bin_width, bin_width)
        h1 = plt.hist(ref_data['{}_err'.format(ax)], bins=bins, log=True, normed=True, color='b',
                      alpha=.4, label='MSF ENAB ({} obs)'.format(len(ref_data)))
        h2 = plt.hist(msd_data['{}_err'.format(ax)], bins=bins, log=True, normed=True, color='r',
                      alpha=.4, label='MSF DISA ({} obs)'.format(len(msd_data)))
        plt.xlabel('{} err (arcsec)'.format(ax))
        plt.legend(loc='upper right', fontsize=7)
        plt.title('99th percentile {} error magnitude (per obs)'.format(ax, i),
                  fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, '{}_err_hist.png'.format(ax)))


def update(datadir, outdir):
    ref_file = os.path.join(datadir, 'ref_obs_data.dat')
    if os.path.exists(ref_file):
        print "Reading reference data from {}".format(ref_file)
        ref_data = Table.read(ref_file, format='ascii')
    else:
        ref_data = get_obs_table('2015:100', '2016:100', msf='ENAB')
        ref_data.write(ref_file, format='ascii')

    data_file = os.path.join(datadir, 'msd_data.dat')
    if os.path.exists(data_file):
        print "Reading previous data from {}".format(data_file)
        last_data = Table.read(data_file, format='ascii')
        new_data = get_obs_table(last_data[-5]['date'], DateTime(), msf='DISA')
        idx_obsid_old_data = np.flatnonzero(last_data['obsid'] == new_data[0]['obsid'])[0]
        msd_data = vstack([last_data[0:idx_obsid_old_data], new_data])
        msd_data.write(data_file, format='ascii')
        # but only use the last year for making these plots
        msd_data = msd_data[msd_data['date'] >= (DateTime() - 365).date]
    else:
        msd_data = get_obs_table(DateTime() - 365, DateTime(), msf='DISA')
        msd_data.write(data_file, format='ascii')

    print msd_data[-1]['date']

    # Filter known bad obsids (50702 test fire)
    for obsid in [50702]:
        msd_data = msd_data[msd_data['obsid'] != obsid]

    one_shot_plot(ref_data, msd_data, 'Reference set and Recent data', outdir=outdir)
    att_err_time_plots(ref_data, msd_data, outdir=outdir)
    att_err_hist(ref_data, msd_data, outdir=outdir)


if __name__ == '__main__':
    opt = get_options()
    if not os.path.exists(opt.outdir):
        os.makedirs(opt.outdir)
    update(outdir=opt.outdir, datadir=opt.datadir)
