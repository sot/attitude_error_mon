#!/usr/bin/env python


import os
import argparse
import jinja2
import numpy as np
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astropy.table import Table, vstack, Column
from astropy.io import ascii
from Ska.Matplotlib import plot_cxctime
from Chandra.Time import DateTime
from kadi import events
from Ska.engarchive import fetch


FILE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_options():
    parser = argparse.ArgumentParser(
        description="Update attitude error plots")
    parser.add_argument("--outdir",
                        default=".")
    parser.add_argument("--datadir",
                        default=".")
    parser.add_argument("--recent-start")
    opt = parser.parse_args()
    return opt


def get_obs_table(start, stop):
    """
    Make a data table of obsids with one shot magnitudes and att errors

    :param start: start time for dwell range to fetch
    :param stop: stop time for range
    :returns: astropy table of observation data
    """
    manvrs = events.manvrs.filter(start=start, stop=stop, n_dwell__gte=1)
    obs_data = []
    for m in manvrs:
        obs = {}
        try:
            obsid = m.get_obsid()
        except ValueError:
            obsid = 0
        if obsid is None:
            obsid = 0

        obs['obsid'] = obsid
        obs['date'] = m.start
        obs['time'] = DateTime(m.start).secs
        obs['manvr_angle'] = m.angle
        obs['one_shot'] = m.one_shot
        obs['one_shot_yaw'] = m.one_shot_yaw
        obs['one_shot_pitch'] = m.one_shot_pitch
        obs['dwell_duration'] = DateTime(m.next_nman_start).secs - DateTime(m.npnt_start).secs

        all_err = {}
        for err_name, err_msid in zip(['roll_err', 'pitch_err', 'yaw_err'],
                                      ['AOATTER1', 'AOATTER2', 'AOATTER3']):
            err = fetch.Msid(err_msid,
                             DateTime(m.npnt_start).secs + 500,
                             m.get_next().start)
            events.dumps.interval_pad = (0, 300)
            err.remove_intervals(events.dumps)
            events.tsc_moves.interval_pad = (0, 300)
            err.remove_intervals(events.tsc_moves)
            err.remove_intervals(events.dark_cal_replicas)
            if len(err.times):
                err.remove_intervals(events.ltt_bads)
            all_err[err_name] = err

        if not len(all_err['roll_err'].vals):
            continue
        obs['roll_err'] = np.degrees(np.percentile(np.abs(all_err['roll_err'].vals), 99)) * 3600
        point_err = np.sqrt((all_err['pitch_err'].vals ** 2) + (all_err['yaw_err'].vals ** 2))
        obs['point_err'] = np.degrees(np.percentile(point_err, 99)) * 3600
        obs_data.append(obs)

    return Table(obs_data)


def one_shot_plot(ref_data, recent_data, outdir='.'):

    # Grab the date of the start and stop of the two data sets.
    d0_str = ref_data['date'][0][0:8]
    d1_str = ref_data['date'][-1][0:8]
    d2_str = recent_data['date'][0][0:8]
    d3_str = recent_data['date'][-1][0:8]

    plt.figure(figsize=(7, 4))
    plt.plot(ref_data['manvr_angle'], ref_data['one_shot'], 'b+',
             markersize=5, markeredgewidth=1.0, alpha=.25, label=f'{d0_str} to {d1_str}')
    plt.plot(recent_data['manvr_angle'], recent_data['one_shot'], 'rx',
             markersize=5, markeredgewidth=.8, label=f'{d2_str} to {d3_str}')
    plt.grid()
    plt.xlim(0, 185)
    plt.ylim(ymin=0)
    plt.ylabel('One Shot (arcsec)')
    plt.xlabel('Manvr Angle (deg)')
    plt.title('One Shot Magnitude', fontsize=12, y=1.05)
    plt.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'one_shot_vs_angle.png'))


def att_err_time_plots(ref_data, recent_data, min_dwell_time=1000, outdir='.'):
    ref_data = ref_data[ref_data['dwell_duration'] >= min_dwell_time]
    recent_data = recent_data[recent_data['dwell_duration'] >= min_dwell_time]

    # Grab the date of the start and stop of the two data sets.
    d0_str = ref_data['date'][0][0:8]
    d1_str = ref_data['date'][-1][0:8]
    d2_str = recent_data['date'][0][0:8]
    d3_str = recent_data['date'][-1][0:8]

    for i, ax in enumerate(['roll', 'point'], 1):
        default_ylims = [0, 1.4]
        if ax == 'roll':
            default_ylims = [0, 12]
        fig = plt.figure(figsize=(5, 3.5))
        ax1 = fig.add_axes([.15, .55, .8, .37])
        plot_cxctime(ref_data['time'], ref_data[f'{ax}_err'], 'b+', markersize=5,
                     alpha=.5)
        plt.ylabel(f'{d0_str} to {d1_str}\n{ax} err (arcsec)', fontsize=9)
        plt.grid()
        plt.margins(x=.1, y=.25)
        ax2 = fig.add_axes([.15, .1, .8, .37])
        plot_cxctime(recent_data['time'], recent_data[f'{ax}_err'], 'rx', markersize=5)
        plt.suptitle(f'99th percentile {ax} error magnitude (per obs)',
                     fontsize=12)
        plt.grid()
        plt.ylabel(f'{d2_str} to {d3_str}\n{ax} err (arcsec)', fontsize=9)
        plt.margins(x=.1, y=.25)
        ylims = plt.ylim()
        setlims = plt.ylim(np.min([ylims[0], default_ylims[0]]),
                           np.max([ylims[1], default_ylims[1]]))
        ax1.set_ylim(setlims)
        plt.setp(ax1.get_xticklabels(), visible=True)
        plt.setp(ax1.get_xticklabels(), fontsize=7)
        plt.setp(ax2.get_xticklabels(), fontsize=7)
        plt.setp(ax1.get_yticklabels(), fontsize=7)
        plt.setp(ax2.get_yticklabels(), fontsize=7)
        plt.setp(ax1.get_xticklabels(), rotation=0)
        plt.setp(ax1.get_xticklabels(), horizontalalignment='center')
        plt.savefig(os.path.join(outdir, f'{ax}_err_vs_time.png'))


def att_err_hist(ref_data, recent_data, label=None, min_dwell_time=1000, outdir='.'):
    ref_data = ref_data[ref_data['dwell_duration'] >= min_dwell_time]
    recent_data = recent_data[recent_data['dwell_duration'] >= min_dwell_time]

    # Grab the date of the start and stop of the two data sets.
    d0_str = ref_data['date'][0][0:8]
    d1_str = ref_data['date'][-1][0:8]
    d2_str = recent_data['date'][0][0:8]
    d3_str = recent_data['date'][-1][0:8]

    for i, ax in enumerate(['roll', 'point'], 1):
        plt.figure(figsize=(5, 3.5))
        bin_width = .05
        lim = np.max([1.4, np.max(recent_data['point_err'])])
        if ax == 'roll':
            bin_width = .25
            lim = np.max([12, np.max(recent_data['roll_err'])])
        bins = np.arange(0, lim + bin_width, bin_width)
        plt.hist(ref_data[f'{ax}_err'], bins=bins, log=True, normed=True, color='b',
                 alpha=.4, label=f'{d0_str} to {d1_str}')
        plt.hist(recent_data[f'{ax}_err'], bins=bins, log=True, normed=True, color='r',
                 alpha=.4, label=f'{d2_str} to {d3_str}')
        plt.xlabel(f'{ax} err (arcsec)')
        plt.legend(loc='upper right', fontsize=7)
        plt.title(f'99th percentile {ax} error magnitude (per obs)',
                  fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'{ax}_err_hist.png'))


def update_file_data(data_file, start, stop):
    if os.path.exists(data_file):
        print(f"Reading previous data from {data_file}")
        last_data = Table.read(data_file, format='ascii')
        new_data = get_obs_table(last_data[-5]['date'], stop)
        if new_data['date'][0] > last_data['date'][-1]:
            data = vstack([last_data, new_data])
            data.sort('date')
        else:
            idx_old_data = np.flatnonzero(last_data['date'] >= new_data['date'][0])[0]
            data = vstack([last_data[0:idx_old_data], new_data])
            data.sort('date')
    else:
        data = get_obs_table(start, stop)
        data.sort('date')
    data.write(data_file, format='ascii', overwrite=True)
    return data


def update(datadir, outdir, full_start, recent_start,
           point_lim=20, roll_lim=40):

    ref_file = os.path.join(datadir, 'ref_obs_data.dat')
    ref_data = update_file_data(ref_file, full_start, recent_start)
    data_file = os.path.join(datadir, 'recent_data.dat')
    recent_data = update_file_data(data_file, recent_start, DateTime())

    one_shot_plot(ref_data, recent_data, outdir=outdir)

    comments_file = os.path.join(outdir, 'comments.dat')
    if os.path.exists(comments_file):
        comments = ascii.read(comments_file, Reader=ascii.FixedWidthTwoLine)
        # convert to a dict
        comments = {comment['obsid']: comment['comment'] for comment in comments}
    else:
        comments = {}

    ok = (recent_data['point_err'] < point_lim) | (recent_data['roll_err'] < roll_lim)
    outliers = recent_data[~ok]
    comments_col = Column(data=[comments.get(row['obsid'], '') for row in outliers],
                          name='comment')
    outliers.add_column(comments_col)
    recent_data = recent_data[ok]

    ok = (ref_data['point_err'] < point_lim) | (ref_data['roll_err'] < roll_lim)
    ref_data = ref_data[ok]
    att_err_time_plots(ref_data, recent_data, outdir=outdir)
    att_err_hist(ref_data, recent_data, outdir=outdir)

    template_html = open(os.path.join(FILE_DIR, 'index_template.html')).read()
    template = jinja2.Template(template_html)
    out_html = template.render(outliers=outliers)
    with open(os.path.join(opt.outdir, 'index.html'), 'w') as fh:
        fh.write(out_html)


if __name__ == '__main__':
    opt = get_options()
    if not os.path.exists(opt.outdir):
        os.makedirs(opt.outdir)

    # Set start of time ranges for data
    if opt.recent_start is None:
        recent_start = DateTime(-60)
    else:
        recent_start = DateTime(opt.recent_start)

    update(outdir=opt.outdir, datadir=opt.datadir,
           full_start=recent_start - 365, recent_start=recent_start)
