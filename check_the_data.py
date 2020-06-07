from tqdm import tqdm
import pathlib
import pandas as pd
import json

IDs = ['20', '21', '23', '25', '29', '30']


def create_csv_from_json_files():  # one time execution
    for id in IDs:
        with open('/mnt/FS1/copd-data/speech/audio_files/copdpatient' + id + '/noNoise/segments.json') as f:
            data = json.load(f)

        df = pd.DataFrame.from_records(data)
        df.drop(columns=['file'], inplace=True)
        df['source'] = df['source'].str[35:]
        df = df[['start_merged', 'end_merged', 'start_src', 'end_src', 'source']]
        df.to_csv(LABELS_PATH + 'segments' + id + '.csv')


def detect_silence_labels(df, filename):  # this function will be executed for each audio file
    starts = df['start'].tolist()  # all the starting times
    ends = df['end'].tolist()  # all the ending times
    num = len(starts)  # how many rows
    missing_starts, missing_ends, duration = [], [], []
    if starts[0] != 0:
        missing_starts.append(0)
        missing_ends.append(starts[0])
        duration.append(starts[0])
    for i in range(0, num - 1):
        missing_starts.append(ends[i])
        missing_ends.append(starts[i + 1])
        duration.append(starts[i + 1] - ends[i])

    if ends[-1] < 120:
        missing_starts.append(ends[-1])
        missing_ends.append(120)
        duration.append(120 - ends[-1])

    newdf = pd.DataFrame()
    newdf['start'] = missing_starts
    newdf['end'] = missing_ends
    newdf['duration'] = duration
    newdf['label'] = 'silence'
    newdf['src_file'] = filename

    return newdf


def append_non_speech_labels(df, filename):  # this function will be executed for each audio file
    df = df.sort_values(by='start')
    starts = df['start'].tolist()  # all the starting times
    ends = df['end'].tolist()  # all the ending times
    num = len(starts)  # how many rows
    missing_starts, missing_ends, duration = [], [], []
    if starts[0] > 0:
        missing_starts.append(0)
        missing_ends.append(starts[0])
        duration.append(starts[0])
    for i in range(0, num - 1):
        missing_starts.append(ends[i])
        missing_ends.append(starts[i + 1])
        duration.append(starts[i + 1] - ends[i])
    if ends[-1] < 120:
        missing_starts.append(ends[-1])
        missing_ends.append(120)
        duration.append(120 - ends[-1])

    newdf = pd.DataFrame()
    newdf['start'] = missing_starts
    newdf['end'] = missing_ends
    newdf['duration'] = duration
    newdf['label'] = 'nonspeech'
    newdf['src_file'] = filename
    # newdf = newdf[newdf['duration'] > 0]  # two rows have wrong labels!

    result = pd.concat([newdf, df])
    return result.sort_values(by='start')


def find_silence_between_2_minutes(partial_df, filename):
    df_silenece = detect_silence_labels(partial_df, filename)
    df_silenece.reset_index(inplace=True, drop=True)  # TODO this line added newly to reset index

    return df_silenece


def find_start_end_merged_from_csv(grp):
    """
    df ['start_merged', 'end_merged', 'src_file']
    """
    df_ranges = pd.DataFrame(columns=['start', 'end', 'src_file'])
    for filename, group in tqdm(grp):
        group_start = group.iloc[0]['start_merged']  # this is for determining where to look for data in txt file
        group_end = group.iloc[-1]['end_merged']
        df2 = {'start': group_start, 'end': group_end, 'src_file': filename}
        df_ranges = df_ranges.append(df2, ignore_index=True)

    return df_ranges


def find_filenames_for_txt_files(df_ranges, txtlines):
    df_txt_with_filename = pd.DataFrame(columns=['start_merged', 'end_merged', 'label', 'src_file'])
    s_index = 0
    counter = 0
    for row in df_ranges.itertuples():
        if counter > len(lines):
            break
        for line in txtlines[s_index:]:
            # print(counter)
            ln = line.strip().split()
            if len(ln) != 3:
                continue
            # print(ln)
            start, end, label = float(ln[0]), float(ln[1]), ln[2]
            if start >= row.start and end <= row.end:  # exactly between the ranges
                df2 = {'start_merged': start, 'end_merged': end, 'label': label, 'src_file': row.src_file}
                df_txt_with_filename = df_txt_with_filename.append(df2, ignore_index=True)
                counter += 1
            elif start <= row.end <= end:  # half of it is in this range
                df2 = {'start_merged': start, 'end_merged': row.end, 'label': label, 'src_file': row.src_file}
                df3 = {'start_merged': row.end, 'end_merged': end, 'label': label,
                       'src_file': df_ranges['src_file'].iloc[[row.Index + 1]].values.item()}
                df_txt_with_filename = df_txt_with_filename.append(df2, ignore_index=True)
                df_txt_with_filename = df_txt_with_filename.append(df3, ignore_index=True)
                counter += 1
            elif start > row.end:  # not in this range anymore
                # counter += 1
                s_index = counter
                break

    # df_txt_with_filename.to_csv('copdpatient20.csv')
    return df_txt_with_filename


IDS = ['21', '23', '25', '29', '30'] # '20',
LABELS_PATH = '/home/tina/research/labels_and_maps/'

# txtfiles = list(pathlib.Path(LABELS_PATH).glob('*.txt'))
# txtfiles.sort()
# mapfiles = list(pathlib.Path(LABELS_PATH).glob('*.csv'))
# mapfiles.sort()
for id in IDS:
    df_maps = pd.read_csv(LABELS_PATH + 'segments' + id + '.csv')
    # print(df_maps['source'].unique())

    df_maps['start_merged'] /= 16000
    df_maps['end_merged'] /= 16000
    df_maps['start_src'] /= 16000
    df_maps['end_src'] /= 16000

    grp_maps = df_maps.groupby(['source'])
    df_ranges = find_start_end_merged_from_csv(grp_maps)

    f = open(LABELS_PATH + 'copdpatient' + id + '.txt', 'r')
    lines = f.readlines()
    df_txt_with_filename = find_filenames_for_txt_files(df_ranges, lines)

    # mapping them
    result_df = pd.DataFrame(columns=['start', 'end', 'duration', 'label', 'src_file'])
    for row in df_txt_with_filename.itertuples():
        s = df_maps[
            (df_maps['start_merged'] <= row.start_merged) & (row.start_merged <= df_maps['end_merged'])].index.values
        e = df_maps[(df_maps['start_merged'] <= row.end_merged) & (row.end_merged <= df_maps['end_merged'])].index.values

        related_df = df_maps.iloc[s[0]:e[-1] + 1, :]
        related_df.reset_index(inplace=True)
        # print(related_df)
        if related_df.shape[0] == 1:  # if it is in a single range thats easy just remap it with considering the difference
            diff = (related_df['start_src'] - related_df['start_merged']).values.item()
            df2 = {'start': row.start_merged + diff, 'end': row.end_merged + diff,
                   'duration': row.end_merged - row.start_merged, 'label': row.label,
                   'src_file': row.src_file}
            result_df = result_df.append(df2, ignore_index=True)
        else:
            for i in range(0, len(related_df)):
                if i == 0:  # first
                    diff = (related_df.iloc[i]['start_src'] - related_df.iloc[i]['start_merged'])
                    if related_df.iloc[i]['end_merged'] - row.start_merged != 0 :
                        df2 = {'start': row.start_merged + diff, 'end': related_df.iloc[i]['end_merged'] + diff,
                               'duration': related_df.iloc[i]['end_merged'] - row.start_merged, 'label': row.label,
                               'src_file': row.src_file}
                        result_df = result_df.append(df2, ignore_index=True)
                elif i == len(related_df) - 1:  # last
                    # print("last")
                    diff = (related_df.iloc[i]['start_src'] - related_df.iloc[i]['start_merged'])
                    if row.end_merged - related_df.iloc[i]['start_merged'] != 0:
                        df2 = {'start': related_df.iloc[i]['start_merged'] + diff, 'end': row.end_merged + diff,
                               'duration': row.end_merged - related_df.iloc[i]['start_merged'], 'label': row.label,
                               'src_file': row.src_file}
                        # print(df2)
                        result_df = result_df.append(df2, ignore_index=True)
                else:  # middle
                    # print("middle")
                    if related_df.iloc[i]['end_src'] - related_df.iloc[i]['start_src']:
                        df2 = {'start': related_df.iloc[i]['start_src'], 'end': related_df.iloc[i]['end_src'],
                               'duration': related_df.iloc[i]['end_src'] - related_df.iloc[i]['start_src'], 'label': row.label,
                               'src_file': row.src_file}
                        result_df = result_df.append(df2, ignore_index=True)
    #


    print(result_df)
    print(result_df['duration'].sum())

    final_result = pd.DataFrame(columns=['start', 'end', 'duration', 'label', 'src_file'])
    howmany = 0
    for filename, group in tqdm(result_df.groupby(['src_file'])):
        # print(group)
        result = append_non_speech_labels(group, filename)
        # print(result)
        final_result = final_result.append(result, ignore_index=True)
        howmany += 1

    print(howmany)
    print(final_result)
    final_result.to_csv(LABELS_PATH + "mapped" + id + '.csv')