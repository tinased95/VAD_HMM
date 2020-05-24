import collections
import contextlib
import sys
import wave
import pandas as pd
import os
import webrtcvad
from tqdm import tqdm

def read_wave(path):
    """Reads a .wav file.

    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:

    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).

    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    counter = 0
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        # sys.stdout.write('1\t' if is_speech else '0\t')
        # sys.stdout.write(str(counter * 0.03))
        # print("\n")
        counter += 1
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                start = ring_buffer[0][0].timestamp
                # sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                end = frame.timestamp + frame.duration
                # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield start, end
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
        end = frame.timestamp + frame.duration
    # sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield start, end


def vad_collector_salaar(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    for frame in frames:
        # sys.stdout.write('1' if vad.is_speech(frame.bytes, sampleRate) else '0')
        if not triggered:
            ring_buffer.append(frame)
            num_voiced = len([f for f in ring_buffer
                              if vad.is_speech(f.bytes, sample_rate)])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                # sys.stdout.write('+(%s)' % (ring_buffer[0].timestamp,))
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append(frame)
            num_unvoiced = len([f for f in ring_buffer
                                if not vad.is_speech(f.bytes, sample_rate)])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield [f for f in voiced_frames]
                ring_buffer.clear()
                voiced_frames = []
                # if triggered:
                # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    # sys.stdout.write('\n')
    if voiced_frames:
        yield [f for f in voiced_frames]


def get_voiced_segments(wav_file, agressiveness):
    audio, sample_rate = read_wave(wav_file)
    frames = list(frame_generator(30, audio, sample_rate))
    vad = webrtcvad.Vad(agressiveness)
    segments = vad_collector_salaar(sample_rate, 30, 300, vad, frames)

    segs = []
    for seg in segments:
        timestamp = float(seg[0].timestamp)
        duration = 0
        for frame in seg:
            duration += frame.duration

        segs.append({"start": int(timestamp * sample_rate), "end": int(timestamp + duration) * sample_rate})
        assert segs[-1]["end"] <= len(audio)/2

    return segs, sample_rate, len(audio)/2


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.

    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


def calculate_vad(filename, aggressiveness):
    df = pd.DataFrame(columns=['label', 'src_start_ts', 'src_end_ts', 'src_file'])
    vad = webrtcvad.Vad(aggressiveness)
    audio, sample_rate = read_wave(filename)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames)
    for i, segment in enumerate(segments):
        df = df.append({'label': 's', 'src_start_ts': segment[0], 'src_end_ts': segment[1], 'src_file': os.path.basename(filename)}
                       , ignore_index=True)

    return df


def main():
    COL_LIST = ["src_start_ts", "src_end_ts", "label", "src_file"]
    df_mapped = pd.read_csv('/mnt/FS1/copd-data/speech/mapped_labels.csv', usecols=COL_LIST)

    basepath = '/mnt/FS1/copd-data/main/features/'
    df = pd.DataFrame(columns=['label', 'src_start_ts', 'src_end_ts', 'src_file'])
    df_all = pd.read_csv('/home/tina/all_files.csv')
    # filenames = df_all['src_file']

    # filenames = df_all['src_file'][~df_all['src_file'].isin(df_mapped['src_file'])]
    # filenames = ['copdpatient30/wav/audio_1491817653115.wav']
    filenames = ['copdpatient23/wav/audio_1490317763916.wav'] # copdpatient21/wav/audio_1480460148641.wav
    for filename in tqdm(filenames):
        df_partial = calculate_vad(basepath + filename, 3)
        df = df.append(df_partial, ignore_index=True)

    # for filename in tqdm(filenames):
    #     segs, samplerate, len = get_voiced_segments(basepath + filename, 3) # basepath + filename
    #     for seg in segs:
    #         df1 = pd.DataFrame(columns=['label', 'src_start_ts', 'src_end_ts', 'src_file'])
    #         df1.loc[0] = ['s', seg['start']/16000, seg['end']/16000, filename]
    #         df = df.append(df1, ignore_index=True)

    print(df)
    # df.to_csv('sileneces_0.csv')

main()
