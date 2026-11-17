#!/usr/bin/env python3
import datetime
import pytz
import pytest
import opensoundscape
import opensoundscape.data_selection as selection
import pandas as pd
import shutil
from pathlib import Path
from opensoundscape import audio


@pytest.fixture
def upsample_df():
    return pd.read_csv("tests/csvs/to_upsample.csv")


@pytest.fixture
def resample_df():
    return pd.read_csv("tests/csvs/to_resample.csv").set_index("file")


@pytest.fixture
def multihot_multiindex_df():
    files = "aabbccccccccccccccc"
    files = list(files)
    return pd.DataFrame(
        {
            "file": files,
            "start_time": list(range(len(files))),
            "end_time": list(range(1, len(files) + 1)),
            "class1": [1] * len(files),
            "class2": [1] * len(files),
        }
    ).set_index(["file", "start_time", "end_time"])


@pytest.fixture
def multihot_multiindex_df_with_duplicates():
    files = "aabbccccccccccccccc"
    files = list(files)
    return pd.DataFrame(
        {
            "file": files,
            "start_time": [0] * len(files),
            "end_time": [2] * len(files),
            "class1": [1] * len(files),
            "class2": [1] * len(files),
        }
    ).set_index(["file", "start_time", "end_time"])


@pytest.fixture
def input_dataframe():
    return pd.DataFrame(
        {
            "Labels": [
                "foo",
                "bar",
                "baz",
                "foo",
                "bar",
                "baz",
                "foo",
                "bar",
                "baz",
                "foo",
                "bar",
                "baz",
                "foo",
                "bar",
                "baz",
            ]
        }
    )


def test_resample_basic(resample_df):
    # changed behavior: now retains all-0 rows by default
    for _ in range(5):
        df = selection.resample(resample_df, 2)
        assert df.shape[0] == 2 * 3 + 2


def test_resample_no_upsample(resample_df):
    for _ in range(5):
        df = selection.resample(resample_df, 2, upsample=False)
        assert df.shape[0] == 5 + 2


def test_resample_no_downsample(resample_df):
    for _ in range(5):
        df = selection.resample(resample_df, 2, downsample=False)
        assert df.shape[0] == 8 + 2


def test_resample_inclue_negatives(resample_df):
    for negatives in (0, 1, 5):
        df = selection.resample(
            resample_df, 1, n_samples_without_labels=negatives, downsample=True
        )
        assert df.shape[0] == 3 + negatives


def test_upsample_basic(upsample_df):
    for _ in range(100):
        upsampled_df = selection.upsample(upsample_df)
        assert upsampled_df.shape[0] == 20


def test_resample_no_negatives(resample_df):
    for _ in range(5):
        df = selection.resample(resample_df, 2, n_samples_without_labels=0)
        assert df.shape[0] == 2 * 3


def test_train_test_split(multihot_multiindex_df):
    # label df with (file,start_time,end_time) multi-index

    # split by file
    train_df, test_df = selection.train_test_split(
        multihot_multiindex_df, test_size=0.5, random_state=42, by_file=True
    )
    # make sure all rows with the same file name are in the same set
    train_files = set(train_df.index.get_level_values(0))
    test_files = set(test_df.index.get_level_values(0))
    assert train_files.isdisjoint(test_files)

    # split by row
    train_df, test_df = selection.train_test_split(
        multihot_multiindex_df, test_size=0.5, random_state=42, by_file=False
    )
    # make sure rows with the same file name can be in different sets
    trial = 0
    while trial < 50:
        train_df, test_df = selection.train_test_split(
            multihot_multiindex_df, test_size=0.5, random_state=trial, by_file=False
        )
        train_files = set(train_df.index.get_level_values(0))
        test_files = set(test_df.index.get_level_values(0))
        if not train_files.isdisjoint(test_files):
            break  # found a trial where files are in both sets
        trial += 1


def test_warns_data_leakage(multihot_multiindex_df_with_duplicates):
    with pytest.warns(UserWarning, match="duplicate"):
        selection.train_test_split(
            multihot_multiindex_df_with_duplicates,
            test_size=0.5,
            random_state=42,
            by_file=False,
        )


@pytest.mark.parametrize("overnight", [False, True])
def test_select_recording_start_times(monkeypatch, overnight):
    starts = {
        "a": datetime.datetime(2024, 1, 2, 1, tzinfo=datetime.timezone.utc),
        "b": datetime.datetime(2024, 1, 2, 12, tzinfo=datetime.timezone.utc),
    }
    monkeypatch.setattr(
        audio, "parse_metadata", lambda f: {"recording_start_time": starts[f]}
    )
    bounds = (
        (datetime.time(22), datetime.time(2))
        if overnight
        else (datetime.time(10), datetime.time(14))
    )
    assert selection.select_files_in_time_range(iter(["a", "b", "a"]), *bounds, "UTC") == (
        ["a", "a"] if overnight else ["b"]
    )
    assert selection.select_files_in_date_range(
        ["a", "b"],
        datetime.date(2024, 1, 1),
        datetime.date(2024, 1, 1),
        "America/New_York",
    ) == ["a"]


def test_select_missing_and_filename_fallback(monkeypatch):
    monkeypatch.setattr(audio, "parse_metadata", lambda f: {})
    args = (datetime.date(2024, 1, 1), datetime.date(2024, 1, 1), "UTC")
    with pytest.raises(ValueError):
        selection.select_files_in_date_range(
            ["20240101_120000.WAV"], *args, raise_exceptions=True
        )
    assert (
        selection.select_files_in_date_range(["unknown"], *args, raise_exceptions=False)
        == []
    )
    assert selection.select_files_in_date_range(
        ["20240101_120000.WAV"],
        *args,
        parse_timestamp_from_filename=True,
        file_timezone="UTC",
    ) == ["20240101_120000.WAV"]


def test_select_timezone_conflict(monkeypatch):
    monkeypatch.setattr(
        audio,
        "parse_metadata",
        lambda f: {
            "recording_start_time": datetime.datetime(
                2024, 1, 1, tzinfo=datetime.timezone.utc
            )
        },
    )
    with pytest.raises(selection.GetRecordingStartTimeError):
        selection.select_files_in_time_range(
            ["a"],
            datetime.time(0),
            datetime.time(23),
            "UTC",
            file_timezone="America/New_York",
            raise_exceptions=True,
        )


@pytest.mark.parametrize("raise_exceptions", [False, True])
@pytest.mark.parametrize(
    "hour",
    [datetime.datetime(2024, 3, 10, 2, 30), datetime.datetime(2024, 11, 3, 1, 30)],
)
def test_select_dst_failure(monkeypatch, raise_exceptions, hour):
    monkeypatch.setattr(
        audio, "parse_metadata", lambda f: {"recording_start_time": hour}
    )
    options = dict(file_timezone="America/New_York", raise_exceptions=raise_exceptions)
    if raise_exceptions:
        with pytest.raises(selection.GetRecordingStartTimeError) as error:
            selection.select_files_in_time_range(
                ["a"], datetime.time(0), datetime.time(23), "UTC", **options
            )
        assert isinstance(
            error.value.__cause__, (pytz.AmbiguousTimeError, pytz.NonExistentTimeError)
        )
    else:
        assert (
            selection.select_files_in_time_range(
                ["a"], datetime.time(0), datetime.time(23), "UTC", **options
            )
            == []
        )


def test_select_validates_empty_input():
    with pytest.raises(pytz.UnknownTimeZoneError):
        selection.select_files_in_time_range(
            [], datetime.time(0), datetime.time(23), "invalid"
        )


@pytest.fixture()
def metadata_wav_str():
    return "tests/audio/metadata.wav"


@pytest.fixture()
def gps_metadata_wav_str():
    return "tests/audio/audiomoth_gps.wav"


@pytest.fixture()
def no_timestamp_wav_str():
    return "tests/audio/short_no_metadata.wav"


@pytest.mark.parametrize(
    "fixture_name,recording_date,recording_time",
    [
        ("metadata_wav_str", datetime.date(2020, 4, 4), datetime.time(10, 25)),
        ("gps_metadata_wav_str", datetime.date(2024, 5, 28), datetime.time(10)),
    ],
)
def test_select_file_based_metadata(
    request, fixture_name, recording_date, recording_time
):
    file = request.getfixturevalue(fixture_name)
    assert selection.select_files_in_date_range(
        [file], recording_date, recording_date, "UTC", raise_exceptions=True
    ) == [file]
    other_date = recording_date + datetime.timedelta(days=1)
    assert (
        selection.select_files_in_date_range(
            [file], other_date, other_date, "UTC", raise_exceptions=True
        )
        == []
    )
    assert selection.select_files_in_time_range(
        [file], recording_time, recording_time, "UTC", raise_exceptions=True
    ) == [file]
    assert (
        selection.select_files_in_time_range(
            [file], datetime.time(12), datetime.time(13), "UTC", raise_exceptions=True
        )
        == []
    )


def test_select_file_based_metadata_across_midnight(metadata_wav_str):
    file = metadata_wav_str
    assert selection.select_files_in_date_range(
        [file],
        datetime.date(2020, 4, 5),
        datetime.date(2020, 4, 5),
        "Pacific/Kiritimati",
        raise_exceptions=True,
    ) == [file]
    assert selection.select_files_in_time_range(
        [file],
        datetime.time(22),
        datetime.time(2),
        "Pacific/Kiritimati",
        raise_exceptions=True,
    ) == [file]


@pytest.mark.parametrize(
    "selector,bounds",
    [
        (
            opensoundscape.select_files_in_date_range,
            (datetime.date(2020, 4, 4), datetime.date(2020, 4, 4), "UTC"),
        ),
        (
            opensoundscape.select_files_in_time_range,
            (datetime.time(10), datetime.time(11), "UTC"),
        ),
    ],
)
def test_select_file_based_missing_timestamp(
    metadata_wav_str, no_timestamp_wav_str, selector, bounds
):
    files = [metadata_wav_str, no_timestamp_wav_str, Path(metadata_wav_str)]
    assert selector(files, *bounds, raise_exceptions=False) == [files[0], files[2]]
    with pytest.raises(
        opensoundscape.GetRecordingStartTimeError, match=no_timestamp_wav_str
    ) as error:
        selector(files, *bounds)
    assert isinstance(error.value.__cause__, ValueError)


def test_select_file_based_filename_fallback(tmp_path, no_timestamp_wav_str):
    file = tmp_path / "20240101_120000.WAV"
    shutil.copyfile(no_timestamp_wav_str, file)
    bounds = (datetime.date(2024, 1, 1), datetime.date(2024, 1, 1), "UTC")
    assert (
        selection.select_files_in_date_range([file], *bounds, raise_exceptions=False)
        == []
    )
    assert selection.select_files_in_date_range(
        [file],
        *bounds,
        parse_timestamp_from_filename=True,
        file_timezone="UTC",
        raise_exceptions=True,
    ) == [file]
    with pytest.raises(selection.GetRecordingStartTimeError) as error:
        selection.select_files_in_date_range(
            [file],
            *bounds,
            parse_timestamp_from_filename=True,
            raise_exceptions=True,
        )
    assert "file_timezone" in str(error.value.__cause__)


def test_select_file_based_metadata_precedes_filename(tmp_path, metadata_wav_str):
    file = tmp_path / "20240101_120000.WAV"
    shutil.copyfile(metadata_wav_str, file)
    assert selection.select_files_in_date_range(
        [file],
        datetime.date(2020, 4, 4),
        datetime.date(2020, 4, 4),
        "UTC",
        parse_timestamp_from_filename=True,
        raise_exceptions=True,
    ) == [file]
