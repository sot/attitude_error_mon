#!/usr/bin/env python

import argparse
from pathlib import Path

import astropy.units as u
import jinja2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Column, Table, vstack
from cheta import fetch
from cxotime import CxoTime
from kadi import events
from ska_matplotlib import plot_cxctime

FILE_DIR = Path(__file__).parent
ROLL_LIM = 20
POINT_LIM = 10


def get_options():
    parser = argparse.ArgumentParser(description="Update attitude error plots")
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--datadir", default=".")
    parser.add_argument(
        "--recent-start",
        help="Date to use as reference for highlighted/recent data,"
        + " default is to highlight last 60 days",
    )
    parser.add_argument(
        "--maude",
        help="Use MAUDE for telemetry",
        action="store_true",
    )
    return parser


def get_obs_table(start, stop, use_maude=False):
    """
    Make a data table of obsids with one shot magnitudes and att errors

    Parameters
    ----------
    start : CxoTimeLike
        Start time for the data
    stop : CxoTimeLike
        Stop time for the data
    use_maude : bool
        Use MAUDE for telemetry, default is to use CXC

    Returns
    -------
    Table
        Table of obsids with one shot magnitudes and 99th percentile att errors
    """

    # Set up to process
    if use_maude:
        manvrs = events.manvrs.filter(start=start, stop=stop)
        cheta_data_source = "maude allow_subset=False"
    else:
        _, atter1_end = fetch.get_time_range("AOATTER1", format="date")
        manvrs = events.manvrs.filter(start=start, next_nman_start__lte=atter1_end)
        cheta_data_source = "cxc"

    obs_data = []
    last_npnt_stop = None
    for m in manvrs:
        obs = {}
        try:
            obsid = m.get_obsid()
        except ValueError:
            obsid = 0
        if obsid is None:
            obsid = 0

        if last_npnt_stop is not None and m.npnt_start is not None:
            obs["nmm_time"] = CxoTime(m.npnt_start).secs - CxoTime(last_npnt_stop).secs
        else:
            obs["nmm_time"] = 0
        if m.npnt_stop is not None:
            last_npnt_stop = m.npnt_stop

        if m.npnt_start is None or obs["nmm_time"] == 0:
            continue

        obs["obsid"] = obsid
        obs["date"] = m.start
        obs["time"] = CxoTime(m.start).secs
        obs["manvr_angle"] = m.angle
        obs["one_shot"] = m.one_shot
        obs["one_shot_yaw"] = m.one_shot_yaw
        obs["one_shot_pitch"] = m.one_shot_pitch
        obs["dwell_duration"] = (
            CxoTime(m.next_nman_start).secs - CxoTime(m.npnt_start).secs
        )

        try:
            all_err = {}
            for err_name, err_msid in zip(
                ["roll_err", "pitch_err", "yaw_err"],
                ["AOATTER1", "AOATTER2", "AOATTER3"],
            ):
                with fetch.data_source(cheta_data_source):
                    err = fetch.Msid(
                        err_msid, CxoTime(m.npnt_start).secs + 500, m.next_nman_start
                    )
                if len(err.times):
                    events.dumps.interval_pad = (0, 300)
                    err.remove_intervals(events.dumps)
                    events.tsc_moves.interval_pad = (0, 300)
                    err.remove_intervals(events.tsc_moves)
                    err.remove_intervals(events.ltt_bads)
                all_err[err_name] = err

            # If there are no samples left, that is not an error condition, and the
            # attitude errors should just be counted as 0.
            if len(all_err["roll_err"].vals) == 0:
                obs["roll_err"] = 0
                obs["point_err"] = 0
            else:
                obs["roll_err"] = (
                    np.degrees(np.percentile(np.abs(all_err["roll_err"].vals), 99))
                    * 3600
                )
                point_err = np.sqrt(
                    (all_err["pitch_err"].vals ** 2) + (all_err["yaw_err"].vals ** 2)
                )
                obs["point_err"] = np.degrees(np.percentile(point_err, 99)) * 3600

        # If there are issues indexing into the AOATTER data, that's an error condition
        # that should just be captured by a large/bogus value.
        except IndexError:
            obs["point_err"] = 999
            obs["roll_err"] = 999
        obs_data.append(obs)

    return Table(obs_data)


def one_shot_plot(ref_data, recent_data, outdir="."):
    # Grab the date of the start and stop of the two data sets.
    d0_str = ref_data["date"][0][0:8]
    d1_str = ref_data["date"][-1][0:8]
    d2_str = recent_data["date"][0][0:8]
    d3_str = recent_data["date"][-1][0:8]

    plt.figure(figsize=(9, 4))
    plt.plot(
        ref_data["manvr_angle"],
        ref_data["one_shot"],
        "b+",
        markersize=5,
        markeredgewidth=1.0,
        alpha=0.25,
        label=f"{d0_str} to {d1_str}",
    )
    plt.plot(
        recent_data["manvr_angle"],
        recent_data["one_shot"],
        "rx",
        markersize=5,
        markeredgewidth=0.8,
        label=f"{d2_str} to {d3_str}",
    )
    plt.grid()
    plt.xlim(-15, 225)
    plt.ylim(ymin=0)
    plt.ylabel("One Shot (arcsec)")
    plt.xlabel("Manvr Angle (deg)")
    plt.title("One shot size vs. manvr angle", fontsize=12, y=1.05)
    plt.legend(loc="upper left", fontsize=8)
    plt.savefig(outdir / "one_shot_vs_angle.png")

    plt.figure(figsize=(9, 4))
    plt.plot(
        ref_data["nmm_time"],
        ref_data["one_shot"],
        "b+",
        markersize=5,
        markeredgewidth=1.0,
        alpha=0.25,
        label=f"{d0_str} to {d1_str}",
    )
    plt.plot(
        recent_data["nmm_time"],
        recent_data["one_shot"],
        "rx",
        markersize=5,
        markeredgewidth=0.8,
        label=f"{d2_str} to {d3_str}",
    )
    plt.grid()
    plt.xlim(0, 4000)
    plt.ylim(ymin=0)
    plt.ylabel("One Shot (arcsec)")
    plt.xlabel("Time in NMM (s)")
    plt.title("One shot size vs. NMM time", fontsize=12, y=1.05)
    plt.legend(loc="upper left", fontsize=8)
    plt.savefig(outdir / "one_shot_vs_nmmtime.png")


def att_err_time_plots(ref_data, recent_data, min_dwell_time=1000, outdir="."):
    ref_data = ref_data[ref_data["dwell_duration"] >= min_dwell_time]
    recent_data = recent_data[recent_data["dwell_duration"] >= min_dwell_time]

    # Grab the date of the start and stop of the two data sets.
    d0_str = ref_data["date"][0][0:8]
    d1_str = ref_data["date"][-1][0:8]
    d2_str = recent_data["date"][0][0:8]
    d3_str = recent_data["date"][-1][0:8]

    for i, ax, msid, lim in zip(
        [1, 2],
        ["roll", "point"],
        ["abs(aoatter1)", "rss(aoatter2,aoatter3)"],
        [ROLL_LIM, POINT_LIM],
    ):
        plt.figure(figsize=(5, 3.5))
        plot_cxctime(
            ref_data["time"],
            ref_data[f"{ax}_err"],
            "b+",
            markersize=5,
            alpha=0.5,
            label=f"{d0_str} to {d1_str}",
        )
        plot_cxctime(
            recent_data["time"],
            recent_data[f"{ax}_err"],
            "rx",
            markersize=5,
            alpha=0.5,
            label=f"{d2_str} to {d3_str}",
        )
        plt.margins(x=0.05, y=0.05)
        plt.ylabel(f"{ax} err (arcsec)", fontsize=9)
        plt.title(f"per obs 99th percentile {ax} err\n ({msid})", fontsize=12)
        plt.grid()
        plt.ylim([0, lim])
        plt.xticks(fontsize=7)
        plt.tight_layout()
        plt.savefig(outdir / f"{ax}_err_vs_time.png")


def att_err_hist(ref_data, recent_data, label=None, min_dwell_time=1000, outdir="."):
    ref_data = ref_data[ref_data["dwell_duration"] >= min_dwell_time]
    recent_data = recent_data[recent_data["dwell_duration"] >= min_dwell_time]

    # Grab the date of the start and stop of the two data sets.
    d0_str = ref_data["date"][0][0:8]
    d1_str = ref_data["date"][-1][0:8]
    d2_str = recent_data["date"][0][0:8]
    d3_str = recent_data["date"][-1][0:8]

    for i, ax, msid in zip(
        [1, 2], ["roll", "point"], ["abs(aoatter1)", "rss(aoatter2, aoatter3)"]
    ):
        plt.figure(figsize=(5, 3.5))
        if ax == "roll":
            bin_width = 0.25
            lim = ROLL_LIM
        else:
            bin_width = 0.05
            lim = POINT_LIM
        bins = np.arange(0, lim + bin_width, bin_width)
        plt.hist(
            ref_data[f"{ax}_err"],
            bins=bins,
            log=True,
            density=True,
            color="b",
            alpha=0.4,
            label=f"{d0_str} to {d1_str}",
        )
        plt.hist(
            recent_data[f"{ax}_err"],
            bins=bins,
            log=True,
            density=True,
            color="r",
            alpha=0.4,
            label=f"{d2_str} to {d3_str}",
        )
        plt.xlabel(f"{ax} err (arcsec)")
        plt.legend(loc="upper right", fontsize=7)
        plt.title(f"per obs 99th percentile {ax} err\n({msid})", fontsize=12)
        plt.grid()
        plt.tight_layout()
        plt.savefig(outdir / f"{ax}_err_hist.png")


def update_file_data(data_file, start, stop, use_maude=False):
    """
    Update the data file with new data

    Parameters
    ----------
    data_file : Path or str
        Path to the data file
    start : CxoTimeLike
        Start time for the data
    stop : CxoTimeLike
        Stop time for the data
    use_maude : bool
        Use MAUDE for telemetry, default is to use CXC
    Returns
    -------
    Table
        Table of obsids with one shot magnitudes and 99th percentile att errors
    """
    if data_file.exists():
        last_data = Table.read(data_file, format="ascii")
        new_data = get_obs_table(last_data[-5]["date"], stop, use_maude=use_maude)
        if new_data["date"][0] > last_data["date"][-1]:
            data = vstack([last_data, new_data])
        else:
            idx_old_data = np.flatnonzero(last_data["date"] >= new_data["date"][0])[0]
            data = vstack([last_data[0:idx_old_data], new_data])
        data = data[data["date"] >= start.date]
    else:
        data = get_obs_table(start, stop, use_maude=use_maude)
    data.sort("date")
    data.write(data_file, format="ascii", overwrite=True)
    return data


def update(datadir, outdir, full_start, recent_start, use_maude=False):
    """
    Update the attitude error plots
    Parameters
    ----------
    datadir : Path
        Path to the data directory
    outdir : Path
        Path to the output directory
    full_start : CxoTimeLike
        Start time for the full data (including earliest plotted data)
    recent_start : CxoTimeLike
        Start time for the recent data (which is plotted in red)
    use_maude : bool
        Use MAUDE for telemetry, default is to use CXC

    Returns
    -------
    None
    """

    outdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)
    data_file = datadir / "data.dat"
    dat = update_file_data(data_file, full_start, CxoTime.now(), use_maude=use_maude)
    recent_data = dat[dat["time"] >= recent_start.secs]
    ref_data = dat[dat["time"] < recent_start.secs]

    one_shot_plot(ref_data, recent_data, outdir=outdir)

    comments_file = outdir / "comments.dat"
    if comments_file.exists():
        comments = Table.read(comments_file, format="ascii.fixed_width_two_line")
        # convert to a dict
        comments = {comment["obsid"]: comment["comment"] for comment in comments}
    else:
        comments = {}

    ok = (recent_data["point_err"] < POINT_LIM) & (recent_data["roll_err"] < ROLL_LIM)
    outliers = recent_data[~ok]
    comments_col = Column(
        data=[comments.get(row["obsid"], "") for row in outliers], name="comment"
    )
    outliers.add_column(comments_col)

    att_err_time_plots(ref_data, recent_data, outdir=outdir)
    att_err_hist(ref_data, recent_data, outdir=outdir)

    template_html = (FILE_DIR / "data" / "index_template.html").read_text()
    template = jinja2.Template(template_html)
    out_html = template.render(
        outliers=outliers,
        one_shot_start=ref_data["date"][0],
        roll_lim=ROLL_LIM,
        point_lim=POINT_LIM,
    )
    with open(outdir / "index.html", "w") as fh:
        fh.write(out_html)


def main(args=None):
    matplotlib.use("Agg")
    opt = get_options().parse_args(args)

    # Set start of time ranges for data
    if opt.recent_start is None:
        recent_start = CxoTime.now() - 60 * u.day
    else:
        recent_start = CxoTime(opt.recent_start)

    update(
        outdir=Path(opt.outdir),
        datadir=Path(opt.datadir),
        full_start=recent_start - 365 * u.day,
        recent_start=recent_start,
        use_maude=opt.maude,
    )


if __name__ == "__main__":
    main()
