import os
import sys
import subprocess
import pytest

# Add in path to source scripts
src_path = os.path.join(os.path.abspath(os.pardir), 'src')
sys.path.insert(0, src_path)

from make_reports import main

@pytest.mark.parametrize("cli_params, expected_out",
        [([], "usage:"),
         (["-c", "dummy"], "No procedural command line options were selected"),
         (["--full", "-brief"], "Cannot select both full and brief report. Please select one or the other."),
         (["-b", "-c", "missing.yaml"], "Cannot find config file missing.yaml"),
         (["-b", "-c", "data/config.yaml", "-r", "1234567890"], "Report date has incorrect format:"),
         (["-b", "-c", "data/config.yaml", "-d", "missing.db"], "Cannot find data in database missing.db"),
         (["-b", "-c", "data/config.yaml"], "unable to open database file"),
         (["-b", "-c", "data/empty_config.yaml"], "data/empty_config.yaml, it is empty"),
         (["-b", "-c", "data/missing_config.yaml"], "data/missing_config.yaml is missing a value for 'tsg_meta_file'"),
         (["-b", "-c", "data/corrupted_config.yaml"], "Error in configuration file:"),
        ])
def test_eval(capsys, cli_params, expected_out):
    with pytest.raises(SystemExit):
        main(["make_reports.py"] + cli_params)
    captured = capsys.readouterr()
    assert expected_out in captured.out
