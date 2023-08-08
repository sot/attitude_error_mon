# Licensed under a 3-clause BSD style license - see LICENSE
from setuptools import setup

entry_points = {
    "console_scripts": [
        "attitude_error_mon=attitude_error_mon.att_err_mon:main",
    ]
}

setup(
    name="attitude_error_mon",
    author="Jean Connelly",
    description="Attitude error monitoring and trending",
    author_email="jconnelly@cfa.harvard.edu",
    use_scm_version=True,
    setup_requires=["setuptools_scm", "setuptools_scm_git_archive"],
    zip_safe=False,
    license=(
        "New BSD/3-clause BSD License\nCopyright (c) 2019"
        " Smithsonian Astrophysical Observatory\nAll rights reserved."
    ),
    entry_points=entry_points,
    packages=["attitude_error_mon"],
    package_data={
        "attitude_error_mon": ["data/index_template.html", "task_schedule.cfg"]
    },
)
