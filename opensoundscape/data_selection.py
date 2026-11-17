"""tools for subsetting and resampling collections"""

import itertools
import pandas as pd
import warnings
import datetime
import pytz
import opensoundscape
from aru_metadata_parser.parse import parse_aru_file_start_time


def train_test_split(df, test_size=0.2, random_state=None, by_file=True):
    """split a multi-index label df into train and evaluation (i.e. validation or test) splits by file

    Input dataframes should have a multi-index of (file,start_time,end_time).

    By default, the split is done by file name, so that all rows with the same file name are in the
    same set. This is to prevent data leakage between the sets, since rows with the same file name
    are likely to be similar. Set `by_file=False` to split by row instead, which randomly assigns
    each row to a split.

    Args:
        df: dataframe with multi-hot encoded labels: multi-index is (file,start_time,end_time)
        test_size: proportion of samples to include in the test set
        random_state: passed to np.random calls. If None, random state is not fixed.
        by_file: if True, split by file name (i.e. all rows with the same file name go in the same set)
            If False, split by row (i.e. rows with the same file name can be in different sets)

    Returns:
        train_df: dataframe with multi-hot encoded labels for the training set
        test_df: dataframe with multi-hot encoded labels for the test set
    """
    if by_file:
        # split by file name
        unique_files = df.index.get_level_values(0).unique()
        test_files = unique_files.to_series().sample(
            frac=test_size, random_state=random_state
        )
        train_files = unique_files.difference(test_files)
        train_df = df[df.index.get_level_values(0).isin(train_files)]
        test_df = df[df.index.get_level_values(0).isin(test_files)]
    else:
        # split by row
        # give a warning if there are duplicate (file,start_time,end_time) indices
        if df.index.duplicated().any():
            warnings.warn(
                "There are duplicate (file,start_time,end_time) indices in the input df. "
                "Splitting by row may result in data leakage between train and test sets "
                "(i.e. samples from the same file may end up in both sets)."
            )
        test_df = df.sample(frac=test_size, random_state=random_state)
        train_df = df.drop(test_df.index)

    return train_df, test_df


def resample(
    df,
    n_samples_per_class,
    n_samples_without_labels=None,
    upsample=True,
    downsample=True,
    with_replace=False,
    random_state=None,
):
    """resample a one-hot encoded label df for a target n_samples_per_class

    Returns a new dataframe with duplicated and/or subset rows. Note that the order of samples changes.

    Can enable/disable upsampling (randomly repeating rows) and downsampling (randomly subsetting rows)

    args:
        df: dataframe with one-hot encoded labels: columns are classes, index is sample name/path
        n_samples_per_class: target number of samples per class
        n_samples_without_labels: number of samples with all-0 labels to include in the returned df
            None or integer.
            - [default: None] keeps all of the original df's rows that have all-0 labels.
            - if integer > 0: upsample or downsample as needed from original df to achieve this number
                of rows with all-0 labels
            - if 0: no all-0 labels are included in the returned df
            Note: `upsample` and `downsample` arguments are ignored for generating all-0 label samples.
        upsample: if True, duplicate samples for classes with <n samples to get to n samples
        downsample: if True, randomly sample classis with >n samples to get to n samples
        with_replace: flag to enable sampling of the same row more than once, default False
        random_state: passed to np.random calls. If None, random state is not fixed.


    Note: The algorithm assumes that the label df is single-label.
    If the label df is multi-label, some classes can end up over-represented.

    Note 2: The resulting df will have samples ordered by class label, even if the input df
    had samples in a random order.
    """

    label_counts = df.sum(0)

    if min(label_counts) < 1 and upsample:
        raise ValueError("Cannot upsample when some classes have zero samples")

    if min(df.sum(1)) > 0 and n_samples_without_labels > 0:
        raise ValueError(
            f"Requested {n_samples_without_labels} samples without any labels, but all samples have labels"
        )

    class_dfs = [None] * len(df.columns)
    for idx, unique_label in enumerate(df.columns):
        no_labels_df = df[df[unique_label] == 1]
        n_class_samples = no_labels_df.shape[0]

        if n_class_samples < n_samples_per_class and (not upsample):
            # we don't want to upsample, so just keep these samples
            class_dfs[idx] = no_labels_df
            continue
        if n_class_samples > n_samples_per_class and (not downsample):
            # we don't want to downsample, so just keep all of samples
            class_dfs[idx] = no_labels_df
            continue

        # upsample or downsample as needed to get to n samples
        num_replicates, remainder = divmod(n_samples_per_class, n_class_samples)

        # take a random sample for the "remainder" portion
        # this is the entirety of the new set of n samples if downsampling,
        # and the samples with an 'extra' representation if upsampling
        random_df = no_labels_df.sample(
            n=remainder, replace=with_replace, random_state=random_state
        )

        # if upsampling, repeat all of the samples as many times as necessary
        if num_replicates > 0:
            repeat_df = pd.concat(itertools.repeat(no_labels_df, num_replicates))
            class_dfs[idx] = pd.concat([repeat_df, random_df])
        else:
            class_dfs[idx] = random_df

    # add samples without any labels, if desired (i.e. "negatives")
    if n_samples_without_labels is None:
        # keep all samples (rows) from original df that did not contain any labels
        class_dfs.append(df[df.sum(1) == 0])
    elif n_samples_without_labels > 0:
        # user specified
        no_labels_df = df[df.sum(1) == 0]
        n_negatives = no_labels_df.shape[0]
        num_replicates, remainder = divmod(n_samples_without_labels, n_negatives)

        # should we consider the upsampling and downsampling flags here?
        # We'll ignore them since the user specified n_samples_without_labels exactly
        # if n_negatives < n_samples_without_labels and (not upsample):
        #     # we don't want to upsample, so just keep these samples
        #     class_dfs.append(sub_df)
        # elif n_negatives > n_samples_without_labels and (not downsample):
        #     # we don't want to downsample, so just keep all of samples
        #     class_dfs.append(sub_df)

        random_df = no_labels_df.sample(
            n=remainder, replace=with_replace, random_state=random_state
        )

        # if upsampling, repeat all of the samples as many times as necessary
        if num_replicates > 0:
            repeat_df = pd.concat(itertools.repeat(no_labels_df, num_replicates))
            class_dfs.append(pd.concat([repeat_df, random_df]))
        else:
            class_dfs.append(random_df)
    # (implicit) else: keep 0 samples without any labels

    return pd.concat(class_dfs)


def upsample(input_df, label_column="Labels", with_replace=False, random_state=None):
    """Given a input DataFrame of categorical labels, upsample to maximum value

    Upsampling removes the class imbalance in your dataset. Rows for each label
    are repeated up to `max_count // rows`. Then, we randomly sample the rows
    to fill up to `max_count`.

    The input df is NOT one-hot encoded in this case, but instead contains
    categorical labels in a specified label_columns

    Args:
        input_df: A DataFrame to upsample
        label_column: The column to draw unique labels from
        with_replace flag to enable sampling of the same row more than once, default False
        random_state: Set the random_state during sampling

    Returns:
        df: An upsampled DataFrame
    """

    unique_labels = input_df[label_column].unique()
    label_counts = input_df.groupby(by=label_column)[label_column].count()
    max_count = label_counts.max()

    dfs = [None] * unique_labels.shape[0]
    for idx, unique_label in enumerate(unique_labels):
        sub_df = input_df[input_df[label_column] == unique_label]
        num_replicates, remainder = divmod(max_count, sub_df.shape[0])

        random_df = sub_df.sample(
            n=remainder, replace=with_replace, random_state=random_state
        )

        repeat_df = pd.concat(itertools.repeat(sub_df, num_replicates))

        dfs[idx] = pd.concat([repeat_df, random_df])

    return pd.concat(dfs)


class GetRecordingStartTimeError(ValueError):
    """Raised when a recording start time cannot be resolved in strict mode."""


def _read_recording_start_time(file, parse_timestamp_from_filename):
    """Read a datetime from metadata, optionally falling back to the filename."""
    metadata = opensoundscape.audio.parse_metadata(file) or {}
    recording_start_time = metadata.get("recording_start_time")
    if isinstance(recording_start_time, datetime.datetime):
        return recording_start_time
    # Note, this will always return a naive datetime as we don't use the filename_timezone argument
    if parse_timestamp_from_filename:
        return parse_aru_file_start_time(file)
    raise ValueError("metadata does not contain a recording start timestamp")


def _localize_recording_start_time(recording_start_time, file_timezone):
    """Return an aware datetime, validating any supplied recorder timezone."""
    if recording_start_time.utcoffset() is None:
        if file_timezone is None:
            raise ValueError("provide file_timezone for naive timestamps")
        # is_dst here will raise if daylight saving time makes the timestamp ambiguous or nonexistent
        return file_timezone.localize(recording_start_time, is_dst=None)

    # Note we only ever reach this if recording_start_time is not naive (read from filename)
    # Recording start time is localized, if we have a file timezone we can check their UTC offsets agree
    if file_timezone is not None:
        converted_start_time = recording_start_time.astimezone(file_timezone)
        if converted_start_time.utcoffset() != recording_start_time.utcoffset():
            raise ValueError("file_timezone conflicts with timestamp offset")
    return recording_start_time


def _select_recordings(
    files,
    predicate,
    target_timezone,
    parse_timestamp_from_filename,
    file_timezone,
    raise_exceptions,
):
    """Resolve timestamps, enact exception policy, and filter using predicate function.

    Args:
        files: Iterable of path-like objects; order and duplicates are preserved.
        predicate: Function passed the datetime objects resolved from a file,
        returns a boolean indicating if the file should be included in the filtered
        results
        target_timezone: pytz timezone name in which calendar dates are compared.
        parse_timestamp_from_filename: Attempt aru-metadata-parser filename
        parsing only if metadata lacks a usable timestamp.
        file_timezone: pytz timezone name for naive timestamps. For aware timestamps,
        its offset must agree with the encoded offset at that instant.
        raise_exceptions: If False, skip files with timestamp errors. If True, raise
        GetRecordingStartTimeError with the original cause. Includes missing
        timestamps, DST failures, and timezone conflicts.

    Returns:
        List of original path objects whose datetimes match the predicate function.

    Raises:
        ValueError: Invalid bounds, timezone names, or flags.
        GetRecordingStartTimeError: File timestamp failure when raise_exceptions=True.
    """
    target_timezone = pytz.timezone(target_timezone)
    if file_timezone is not None:
        file_timezone = pytz.timezone(file_timezone)

    selected = []
    for file in files:
        try:
            recording_start_time = _read_recording_start_time(
                file,
                parse_timestamp_from_filename=parse_timestamp_from_filename,
            )
            recording_start_time = _localize_recording_start_time(
                recording_start_time,
                file_timezone=file_timezone,
            )
            recording_start_time = recording_start_time.astimezone(target_timezone)
        except Exception as exc:
            if raise_exceptions:
                raise GetRecordingStartTimeError(f"Exception on file {file}") from exc
            continue
        if predicate(recording_start_time):
            selected.append(file)
    return selected


def select_files_in_date_range(
    files,
    first_date,
    last_date,
    target_timezone,
    parse_timestamp_from_filename=False,
    file_timezone=None,
    raise_exceptions=True,
):
    """Selects recordings with start times within a date range, including start and end date

    Args:
        files: Iterable of path-like objects; order and duplicates are preserved.
        first_date: Inclusive datetime.date bound (not datetime.datetime).
        last_date: Inclusive datetime.date bound, at least first_date.
        target_timezone: pytz timezone name in which calendar dates are compared.
        parse_timestamp_from_filename: Attempt aru-metadata-parser filename parsing
        only if metadata lacks a usable timestamp. Defaults to False.
        file_timezone: pytz timezone name for naive timestamps. For aware timestamps,
        its offset must agree with the encoded offset at that instant.
        raise_exceptions: If True (default), raise GetRecordingStartTimeError
        with the original cause. If False, skip files with timestamp errors.
        Errors include missing timestamps, conflicting timezone offsets,
        naive timestamps without a supplied zone, and ambiguous or
        nonexistent local times during daylight-saving transitions. Invalid
        date/time bounds and unrecognized timezone names raise before
        processing files. File modification times and the computer's local
        timezone are never used.

    Returns:
        List of original path objects whose recording starts match the range.

    Raises:
        ValueError: Invalid bounds, timezone names, or flags.
        GetRecordingStartTimeError: File timestamp failure when raise_exceptions=True.

    Examples:
        >>> from datetime import date
        >>> select_files_in_date_range([], date(2024, 1, 1), date(2024, 1, 2), 'UTC')
        []

    Notes:
        Only recording starts are considered, not recording overlap or file
        modification times. Metadata takes precedence over filenames. If timezone is
        not found in metadata or filename, this will require file_timezone.
        Ambiguous/nonexistent daylight saving times are unresolved (will raise an
        exception or skip). Filename recognition, including fuzzy parsing, is best
        effort.
    """
    if any(
        not isinstance(d, datetime.date) or isinstance(d, datetime.datetime)
        for d in (first_date, last_date)
    ):
        raise ValueError("bounds must be datetime.date objects, not datetimes")
    if first_date > last_date:
        raise ValueError("first_date must be <= last_date")

    def predicate(dt):
        return first_date <= dt.date() <= last_date

    return _select_recordings(
        files,
        predicate,
        target_timezone,
        parse_timestamp_from_filename,
        file_timezone,
        raise_exceptions,
    )


def select_files_in_time_range(
    files,
    earliest_time,
    latest_time,
    target_timezone,
    parse_timestamp_from_filename=False,
    file_timezone=None,
    raise_exceptions=True,
):
    """Select recordings by time of day they started, inclusive of end values

    Args:
        files: Iterable of path-like objects.
        earliest_time: Timezone-free datetime.time lower boundary.
        latest_time: Timezone-free datetime.time upper boundary. If earlier
        than earliest_time, the range wraps midnight. Equal bounds match
        exactly.
        target_timezone: pytz timezone name for wall-clock comparisons.
        parse_timestamp_from_filename: Enable best-effort filename fallback;
        metadata is preferred. Defaults to False.
        file_timezone: Timezone name for naive timestamps, or an offset
        consistency check for aware timestamps; never overrides an encoded
        timezone.
        raise_exceptions: If True (default), raise GetRecordingStartTimeError
        with the original cause. If False, skip files with timestamp errors.
        Errors include missing timestamps, conflicting timezone offsets,
        naive timestamps without a supplied zone, and ambiguous or
        nonexistent local times during daylight-saving transitions. Invalid
        date/time bounds and unrecognized timezone names raise before
        processing files. File modification times and the computer's local
        timezone are never used.

    Returns:
        List of original paths preserving order and duplicates.

    Raises:
        ValueError: Invalid bounds, timezone names, or flags.
        GetRecordingStartTimeError: File timestamp failure when
        raise_exceptions=True.

    Notes:
        Only recording starts are considered, not recording overlap or file
        modification times. Metadata takes precedence over filenames. If timezone is
        not found in metadata or filename, this will require file_timezone.
        Ambiguous/nonexistent daylight saving times are unresolved (will raise an
        exception or skip). Filename recognition, including fuzzy parsing, is best
        effort.
    """
    if any(
        not isinstance(t, datetime.time) or t.tzinfo is not None
        for t in (earliest_time, latest_time)
    ):
        raise ValueError("bounds must be timezone-free datetime.time objects")

    def predicate(dt):
        t = dt.time()
        if earliest_time <= latest_time:
            return earliest_time <= t <= latest_time
        return t >= earliest_time or t <= latest_time

    return _select_recordings(
        files,
        predicate,
        target_timezone,
        parse_timestamp_from_filename,
        file_timezone,
        raise_exceptions,
    )
