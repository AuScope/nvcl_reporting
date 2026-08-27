import os
import sys
import subprocess
import pytest

# Add in path to source scripts
src_path = os.path.join(os.path.abspath(os.pardir), 'src')
sys.path.insert(0, src_path)

my_dir = os.path.dirname(os.path.abspath(__file__))

from make_reports import main

@pytest.mark.parametrize("cli_params, expected_out",
        [([], "usage:"),
         (["-c", "dummy"], "No procedural command line options were selected"),
         (["--full", "-brief"], "Cannot select both full and brief report. Please select one or the other."),
         (["-b", "-c", "missing.yaml"], "Cannot find config file missing.yaml"),
         (["-b", "-c", os.path.join(my_dir, "data/config.yaml"), "-r", "1234567890"], "Report date has incorrect format:"),
         (["-b", "-c", os.path.join(my_dir, "data/empty_config.yaml")], "data/empty_config.yaml, it is empty"),
         (["-b", "-c", os.path.join(my_dir, "data/missing_config.yaml")], "data/missing_config.yaml is missing a value for 'tsg_meta_file'"),
         (["-b", "-c", os.path.join(my_dir, "data/corrupted_config.yaml")], "Error in configuration file:"),
        ])
def test_eval(capsys, cli_params, expected_out):
    os.environ["POSTGRES_DB"] = "dummy"
    os.environ["POSTGRES_USER"] = "dummy"
    os.environ["POSTGRES_PASSWORD"] = "dummy"
    with pytest.raises(SystemExit):
        main(["make_reports.py"] + cli_params)
        captured = capsys.readouterr()
        #with capsys.disabled():
        #    print(f"{captured=}")
        assert(expected_out in captured.out or expected_out in captured.err)
