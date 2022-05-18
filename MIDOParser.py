"""
MIDI Parser
There is no need to use this module directly. All you need is
available in the top level module.
"""
from collections import deque
from .messages import Message
from .tokenizer import Tokenizer


class Parser(object):
    """
    MIDI byte stream parser
    Parses a stream of MIDI bytes and produces messages.
    Data can be put into the parser in the form of
    integers, byte arrays or byte strings.
    """
    def __init__(self, data=None):
        # For historical reasons self.messages is public and must be a
        # deque(). (It is referenced directly inside ports.)
        self.messages = deque()
        self._tok = Tokenizer()
        if data:
            self.feed(data)

    def _decode(self):
        for midi_bytes in self._tok:
            self.messages.append(Message.from_bytes(midi_bytes))

    def feed(self, data):
        """Feed MIDI data to the parser.
        Accepts any object that produces a sequence of integers in
        range 0..255, such as:
            [0, 1, 2]
            (0, 1, 2)
            [for i in range(256)]
            (for i in range(256)]
            bytearray()
            b''  # Will be converted to integers in Python 2.
        """
        self._tok.feed(data)
        self._decode()

    def feed_byte(self, byte):
        """Feed one MIDI byte into the parser.
        The byte must be an integer in range 0..255.
        """
        self._tok.feed_byte(byte)
        self._decode()

    def get_message(self):
        """Get the first parsed message.
        Returns None if there is no message yet. If you don't want to
        deal with None, you can use pending() to see how many messages
        you can get before you get None, or just iterate over the
        parser.
        """
        for msg in self:
            return msg
        else:
            return None

    def pending(self):
        """Return the number of pending messages."""
        return len(self.messages)

    __len__ = pending

    def __iter__(self):
        """Yield messages that have been parsed so far."""
        while len(self.messages) > 0:
            yield self.messages.popleft()


def parse_all(data):
    """Parse MIDI data and return a list of all messages found.
    This is typically used to parse a little bit of data with a few
    messages in it. It's best to use a Parser object for larger
    amounts of data. Also, tt's often easier to use parse() if you
    know there is only one message in the data.
    """
    return list(Parser(data))


def parse(data):
    """ Parse MIDI data and return the first message found.
    Data after the first message is ignored. Use parse_all()
    to parse more than one message.
    """
    return Parser(data).get_message()

import numpy as np
from copy import deepcopy
from scipy.sparse import csc_matrix
import miditoolkit.midi.containers as ct


PITCH_RANGE = 128


# def get_onsets_pianoroll():
#     pass


# def get_offsets_pianoroll():
#     pass


def notes2pianoroll(
        note_stream_ori, 
        ticks_per_beat=480, 
        downbeat=None, 
        resample_factor=1.0, 
        resample_method=round,
        binary_thres=None,
        max_tick=None,
        to_sparse=False, 
        keep_note=True):
    
    # pass by value
    note_stream = deepcopy(note_stream_ori)

    # sort by end time
    note_stream = sorted(note_stream, key=lambda x: x.end)
    
    # set max tick
    if max_tick is None:
        max_tick = 0 if len(note_stream) == 0 else note_stream[-1].end
        
    # resampling
    if resample_factor != 1.0:
        max_tick = int(resample_method(max_tick * resample_factor))
        for note in note_stream:
            note.start = int(resample_method(note.start * resample_factor))
            note.end = int(resample_method(note.end * resample_factor))
    
    # create pianoroll
    time_coo = []
    pitch_coo = []
    velocity = []
    
    for note in note_stream:
        # discard notes having no velocity
        if note.velocity == 0:
            continue

        # duration
        duration = note.end - note.start

        # keep notes with zero length (set to 1)
        if keep_note and (duration == 0):
            duration = 1
            note.end += 1

        # set time
        time_coo.extend(np.arange(note.start, note.end))
        
        # set pitch
        pitch_coo.extend([note.pitch] * duration)
        
        # set velocity
        v_tmp = note.velocity
        if binary_thres is not None:
            v_tmp = v_tmp > binary_thres
        velocity.extend([v_tmp] * duration)
    
    # output
    pianoroll = csc_matrix((velocity, (time_coo, pitch_coo)), shape=(max_tick, PITCH_RANGE))
    pianoroll = pianoroll if to_sparse else pianoroll.toarray()
    
    return pianoroll      


def pianoroll2notes(
        pianoroll,
        resample_factor=1.0):

    binarized = pianoroll > 0
    padded = np.pad(binarized, ((1, 1), (0, 0)), "constant")
    diff = np.diff(padded.astype(np.int8), axis=0)

    positives = np.nonzero((diff > 0).T)
    pitches = positives[0]
    note_ons = positives[1]
    note_offs = np.nonzero((diff < 0).T)[1]

    notes = []
    for idx, pitch in enumerate(pitches):
        st = note_ons[idx] 
        ed = note_offs[idx]
        velocity = pianoroll[st, pitch]
        velocity = max(0, min(127, velocity))
        notes.append(
            ct.Note(
                velocity=int(velocity), 
                pitch=pitch, 
                start=int(st*resample_factor), 
                end=int(ed*resample_factor)))
    notes.sort(key=lambda x: x.start)
    return notes