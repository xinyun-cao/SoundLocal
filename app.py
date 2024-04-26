
import queue
import re
import sys
import time
import signal
import threading
import ctypes
# for opencv
import cv2
import dlib
import numpy as np
from asteroid.models import ConvTasNet
from google.cloud import speech
import pyaudio
import multiprocessing
import samplerate as sr
import math

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech as cloud_speech_types

from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio
import torch
from IPython.display import display, Audio
import torchaudio
from speechbrain.utils.fetching import fetch
from speechbrain.utils.data_utils import split_path
import numpy as np
import time
import wave
import soundfile as sf

from speechbrain.inference.speaker import SpeakerRecognition

# computer microphone
FORMAT    =pyaudio.paFloat32
CHANNELS  =1
FS        =8000
CHUNK     =int(FS/10)
THRESHOLD = 0.01 #silence detection threshold
SILENCE_CHUNKS = 4
STREAMING_LIMIT = 240000 
Device_index = 1
# microphone array
# FORMAT    =pyaudio.paInt32  
# CHANNELS  =16
# fs        =48000
# CHUNK     =int(fs/15)

# Import the base64 encoding library.
import base64

def loadData(path, device, sample_rate):
    savedir = "audio_cache"

    source, fl = split_path(path)
    path = fetch(fl, source=source, savedir=savedir)

    batch, fs_file = torchaudio.load(path)
    batch = batch.to(device)
    fs_model = sample_rate
    if fs_file != fs_model:
        print(
            "Resampling the audio from {} Hz to {} Hz".format(
                fs_file, fs_model
            )
        )
        tf = torchaudio.transforms.Resample(
            orig_freq=fs_file, new_freq=fs_model
        ).to(device)
        batch = batch.mean(dim=0, keepdim=True)
        batch = tf(batch)
    return batch

def sampleData(batch, n, device):
    if batch.shape[1] % n != 0:
        batch = batch[0, :-(batch.shape[1] % n)]
    return batch.view(n, -1).to(device)


class Matcher:
    def __init__(self, device):
        self.speaker1 = torch.empty(size=(0,), device=device)
        self.speaker2 = torch.empty(size=(0,), device=device)
        self.verification = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device":device})
        self.resampler = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000).to(device)
        self.model = separator.from_hparams(
            source="speechbrain/sepformer-libri2mix",
            savedir="pretrained_models/sepformer-libri2mix",
            run_opts={"device":device})

    def addSample(self, audio, speaker):
        if speaker == 1:
            self.speaker1 = torch.cat((self.speaker1, self.resampler(torch.Tensor(audio))))
        elif speaker == 2:
            self.speaker2 = torch.cat((self.speaker2, self.resampler(torch.Tensor(audio))))


    def predict(self, batch):
        est_sources = self.model.separate_batch(batch)
        est_sources = (
            est_sources / est_sources.abs().max(dim=1, keepdim=True)[0]
        )
        return est_sources

    def addSampleBatch(self, batch, speaker):
        for i in range(batch.shape[0]):
            self.addSample(batch[i], speaker)

    

    def match(self, audio1, audio2):
        def fixLength(clip, n):
            while clip.shape[0] < n:
                clip = torch.cat((clip, clip))
            return clip[:n]
        
        s1_temp = self.speaker1
        s2_temp = self.speaker2
        if s1_temp.shape[0] == 0 and s2_temp.shape[0] == 0: # we know nothing, so we guess
            return 0, 1, 0, 0
        elif s1_temp.shape[0] == 0 or s2_temp.shape[0] == 0: # only data about 1 speaker
            if s1_temp.shape[0] == 0:
                s1_temp = torch.zeros_like(s2_temp)
            else:
                s2_temp = torch.zeros_like(s1_temp)
        elif s1_temp.shape[0] != s2_temp.shape[0]: # different amounts of speaker data
            speaklen = max(self.speaker1.shape[0], self.speaker2.shape[0])
            s1_temp = fixLength(s1_temp, speaklen).unsqueeze(0)
            s2_temp = fixLength(s1_temp, speaklen).unsqueeze(0)

        batch1 = torch.vstack((
            s1_temp,
            s1_temp,
            s2_temp,
            s2_temp))

        audio1 = self.resampler(torch.Tensor(audio1)).unsqueeze(0)
        audio2 = self.resampler(torch.Tensor(audio2)).unsqueeze(0)
        batch2 = torch.vstack((
            audio1,
            audio2,
            audio1,
            audio2))

        v = self.verification.verify_batch(batch1, batch2)[0].cpu().numpy()
        if v[0] + v[3] > v[1] + v[2]:
            return 0, 1, v[0] + v[3], v[1] + v[2]
        else:
            return 1, 0, v[1] + v[2], v[0] + v[3]
        
    def assign_speaker(self, batch):
        est_sources = self.predict(batch.unsqueeze(0))
        mv = self.match(est_sources[0, :, 0], est_sources[0, :, 1])        
        return est_sources[0, :, mv[0]], est_sources[0, :, mv[1]]#, m-s, e-m



def get_current_time() -> int:
    """Return Current Time in MS.

    Returns:
        int: Current Time in MS.
    """

    return int(round(time.time() * 1000))

def handler_SIGINT(signum, frame):
    global flag_run_ctrl_c
    flag_run_ctrl_c=False

class ResumableMicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(
        self: object,
        rate: int,
        chunk_size: int,
    ) -> None:
        """Creates a resumable microphone stream.

        Args:
        self: The class instance.
        rate: The audio file's sampling rate.
        chunk_size: The audio file's chunk size.

        returns: None
        """
        self._rate = rate
        self.chunk_size = chunk_size
        self._num_channels = 1
        self._buff = queue.Queue()
        self.closed = True
        self.start_time = get_current_time()
        self.restart_counter = 0
        self.audio_input = []
        self.last_audio_input = []
        self.result_end_time = 0
        self.is_final_end_time = 0
        self.final_request_end_time = 0
        self.bridging_offset = 0
        self.last_transcript_was_final = False
        self.new_stream = True
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

    def __enter__(self: object) -> object:
        """Opens the stream.

        Args:
        self: The class instance.

        returns: None
        """
        self.closed = False
        return self

    def __exit__(
        self: object,
        type: object,
        value: object,
        traceback: object,
    ) -> object:
        """Closes the stream and releases resources.

        Args: 
        self: The class instance.
        type: The exception type.
        value: The exception value.
        traceback: The exception traceback.

        returns: None
        """
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(
        self: object,
        in_data: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Continuously collect data from the audio stream, into the buffer.

        Args:
        self: The class instance.
        in_data: The audio data as a bytes object.
        args: Additional arguments.
        kwargs: Additional arguments.

        returns: None
        """
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self: object) -> object:
        """Stream Audio from microphone to API and to local buffer

        Args:
            self: The class instance.

        returns:
            The data from the audio stream.
        """
        while not self.closed:
            data = []

            if self.new_stream and self.last_audio_input:
                chunk_time = STREAMING_LIMIT / len(self.last_audio_input)

                if chunk_time != 0:
                    if self.bridging_offset < 0:
                        self.bridging_offset = 0

                    if self.bridging_offset > self.final_request_end_time:
                        self.bridging_offset = self.final_request_end_time

                    chunks_from_ms = round(
                        (self.final_request_end_time - self.bridging_offset)
                        / chunk_time
                    )

                    self.bridging_offset = round(
                        (len(self.last_audio_input) - chunks_from_ms) * chunk_time
                    )

                    for i in range(chunks_from_ms, len(self.last_audio_input)):
                        data.append(self.last_audio_input[i])

                self.new_stream = False

            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            self.audio_input.append(chunk)

            if chunk is None:
                return
            data.append(chunk)
            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)

                    if chunk is None:
                        return
                    data.append(chunk)
                    self.audio_input.append(chunk)

                except queue.Empty:
                    break

            yield b"".join(data)


def listen_print_loop(responses: object, stream: object,msg_queue) -> object:
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.

    Arg:
        responses: The responses returned from the API.
        stream: The audio stream to be processed.

    Returns:
        The transcript of the result
    """
    for response in responses:
        if get_current_time() - stream.start_time > STREAMING_LIMIT:
            stream.start_time = get_current_time()
            break

        if not response.results:
            continue

        result = response.results[0]
        
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        result_seconds = 0
        result_micros = 0

        if result.result_end_time.seconds:
            result_seconds = result.result_end_time.seconds

        if result.result_end_time.microseconds:
            result_micros = result.result_end_time.microseconds

        stream.result_end_time = int((result_seconds * 1000) + (result_micros / 1000))

        corrected_time = (
            stream.result_end_time
            - stream.bridging_offset
            + (STREAMING_LIMIT * stream.restart_counter)
        )
        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.

        if result.is_final:
            sys.stdout.write("\033[K")
            sys.stdout.write(str(corrected_time) + ": " + transcript + "\n")

            stream.is_final_end_time = stream.result_end_time
            stream.last_transcript_was_final = True
            
            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                sys.stdout.write("Exiting...\n")
                stream.closed = True
                break
        else:
            sys.stdout.write("\033[K")
            sys.stdout.write(str(corrected_time) + ": " + transcript + "\r")
            msg_queue.put((transcript, time.time()))
            stream.last_transcript_was_final = False
        if (stream.last_transcript_was_final):

            return transcript
            
def read_data(stream):
    """
    Reads data from audio stream & returns 1 x (CHANNELS * CHUNK) np array
    """
    # recording the data
    data = stream.read(CHUNK, exception_on_overflow = True)
    data_sample = np.frombuffer(data, dtype=np.int32)
    data_sample = np.reshape(data_sample, (CHANNELS, CHUNK), order='F')
    # data = np.reshape(data, (-1, CHANNELS))
    return data_sample

def calculate_rms(audio_data):
    # Assuming audio_data is a numpy array of float32 samples
    return np.sqrt(np.mean(np.square(audio_data)))



def loop_mic_v2(flag_run, audio_queue,silence_detected_event,paras):
    p                 =pyaudio.PyAudio()
    info              =p.get_host_api_info_by_index(0)
    numdevices        =info.get('deviceCount')
    target_device_name="nanoSHARC micArray16 UAC2.0"

    # Find target device
    # for i in range(0, numdevices):
    #     if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels'))>0:
    #         tmp=p.get_device_info_by_host_api_device_index(0, i).get('name')
    #         print(tmp)
    #         if tmp.find(target_device_name)>=0:
    #             device_index = i
    #             break

    # Open device
    stream=p.open(format            =paras['FORMAT'],
                  channels          =paras['n_channels'],
                  rate              =paras['fs'],
                  input             =True,
                  frames_per_buffer =paras['n_samples_per_chunk'],
                  #input_device_index=device_index
                  )
    stream.start_stream()

    i = 0
    silence_buffer = []
    silence_count = 0
    while flag_run.value:
        i += 1
        
        #data=read_data(stream) # (# of channels, # of samples)
        data = stream.read(CHUNK)
        audio_queue.put(data)
        #audio_data.extend(data)
        numpy_data = np.frombuffer(data, dtype=np.float32)
        rms_value = calculate_rms(numpy_data)
        
        # Track RMS values
        silence_buffer.append(rms_value)
        if len(silence_buffer) > SILENCE_CHUNKS:
            silence_buffer.pop(0)  # Keep the buffer from growing indefinitely
        
        # Check if all recent chunks are below the silence threshold
        if all(rms < THRESHOLD for rms in silence_buffer):
            silence_count += 1
        else:
            silence_count = 0  # Reset the counter if any chunk is above the threshold
        
        # Determine if silence has been detected for a few consecutive chunks
        if silence_count >= SILENCE_CHUNKS or i == 50:  # Check if maximum chunks or silence threshold reached
                # Notify other process that the speaker may have finished speaking
                silence_detected_event.set()
                silence_count = 0  # Reset silence count after setting the event
                i = 0  # Reset chunk counter
            


            

    # Close all
    stream.stop_stream()
    stream.close()
    p.terminate()

def loop_cam(flag_run, paras):
    PREDICTOR_PATH = r"face_detector/shape_predictor_68_face_landmarks.dat"

    LIP_DIST_CUTOFF = 5.0
    detector = dlib.get_frontal_face_detector()

    cap = cv2.VideoCapture(0)
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    last_message = None
    message_display_time = 0
    display_duration = 3

    while flag_run.value:
        success, image = cap.read()
        h, w = image.shape[:2]

        # detector usage if using dnn
        # blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        # net.setInput(blob)
        # detections = net.forward()
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        # draw bounding box
        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            landmarks = np.matrix([[p.x, p.y]  
                for p in predictor(image, dlib_rect).parts()])
            mouth_point_up = landmarks[62]
            mouth_point_down = landmarks[66]
            # print out landmark for upper lip and lower lip
            counter = 0
            for point in (mouth_point_up, mouth_point_down):  
                pos = (point[0, 0], point[0, 1])
                cv2.circle(image, pos, 2, color=(0, 255, 255), thickness=-1)
                counter+= 1
            dist = np.linalg.norm(mouth_point_up-mouth_point_down)
            #delayed display of captions
            current_time = time.time()
            if dist > LIP_DIST_CUTOFF or (last_message and current_time - message_display_time < display_duration):
                if current_time - message_display_time >= display_duration:
                    try:
                        #last_message, message_display_time = message_queue.get_nowait()
                        print("returned from Google API")
                    except queue.Empty:
                        pass  # No new message, keep displaying the last one

                if last_message:
                    cv2.putText(image, last_message, (x, y - 10), 
                                fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,  
                                fontScale=1,  
                                color=(0, 0, 255))
            else:
                # Reset last_message when not speaking or after duration has passed
                last_message = None

        cv2.imshow("Webcam", image) # This will open an independent window
        if cv2.waitKey(1) & 0xFF==ord('q'): # quit when 'q' is pressed
            cap.release()
            break
    
    cv2.destroyAllWindows() 
    cv2.waitKey(1)
def single_mic(msg_queue):
    """start bidirectional streaming from microphone input to speech API"""
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=FS,
        language_code="en-US",
        max_alternatives=1,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    mic_manager = ResumableMicrophoneStream(FS, CHUNK)
    sys.stdout.write('\nListening, say "Quit" or "Exit" to stop.\n\n')
    sys.stdout.write("End (ms)       Transcript Results/Status\n")
    sys.stdout.write("=====================================================\n")

    with mic_manager as stream:
        while not stream.closed:
            sys.stdout.write(
                "\n" + str(STREAMING_LIMIT * stream.restart_counter) + ": NEW REQUEST\n"
            )

            stream.audio_input = []
            audio_generator = stream.generator()
            # for content in audio_generator:
            #     print(bytes_to_floats(content).shape)

            requests = (
                speech.StreamingRecognizeRequest(audio_content=content)
                for content in audio_generator
            )

            responses = client.streaming_recognize(streaming_config, requests)

            # Now, put the transcription responses to use.
            response = listen_print_loop(responses, stream,msg_queue)
            if stream.result_end_time > 0:
                stream.final_request_end_time = stream.is_final_end_time
            stream.result_end_time = 0
            stream.last_audio_input = []
            stream.last_audio_input = stream.audio_input
            stream.audio_input = []
            stream.restart_counter = stream.restart_counter + 1
            msg_queue.put((response, time.time()))
            with open("example.txt", 'a') as file:
                        # Write new text to the file
                        file.write("stream result:")
                        file.write(response)
                        file.write("\n")
            if not stream.last_transcript_was_final:
                stream.new_stream = True
                sys.stdout.write("\n")
def loop_ML(flag_run, audio_queue, silence_detected_event,msg_queue,speaker1_talking,speaker2_talking,paras):
    client = speech.SpeechClient()
    device = "cpu"
    matcher = Matcher(device) #Matcher(2048, 512, device)
    
    # speaker1file = "audios/JoshOnly.wav"
    # speaker2file = "audios/XinyunOnly.wav"
    # speaker1 = loadData(speaker1file, device, 8000)
    # speaker2 = loadData(speaker2file, device, 8000)
    # speaker1Batch = sampleData(speaker1[:, 1*8000:10*8000], 1, device)
    # speaker2Batch = sampleData(speaker2[:, 1*8000:10*8000], 1, device)
    # matcher.addSampleBatch(speaker1Batch, 1)
    # matcher.addSampleBatch(speaker2Batch, 2)
    j = 1
    while flag_run.value or not audio_queue.empty():
        
        if silence_detected_event.is_set():
            silence_detected_event.clear()
            print("Processing buffered audio...")

            # Collect all available audio chunks
            audio_data = bytearray()
            while not audio_queue.empty():
                audio_chunk = audio_queue.get()
                audio_data.extend(audio_chunk)
            #print("length of audio data received"+str(len(audio_data)))
            # Check if there is any audio to process
            if audio_data:
                audio_array = torch.frombuffer(audio_data, dtype=torch.float32)
                single_speaker = np.frombuffer(audio_data, dtype=np.float32).copy()
                response = []
                if speaker1_talking and speaker2_talking:
                    s1 = None
                    s2 = None
                    s1, s2 = matcher.assign_speaker(audio_array)
                    speaker1_audio = s1.cpu().numpy()
                    speaker2_audio = s2.cpu().numpy()
                    response.append(transcribe_streaming(speaker1_audio, client))
                    response.append(transcribe_streaming(speaker2_audio, client))
                elif speaker1_talking:
                    matcher.addSampleBatch(audio_array.view(1,-1), 1)
                    response.append(transcribe_streaming(single_speaker, client))
                    response.append("")
                elif speaker2_talking:
                    matcher.addSampleBatch(audio_array.view(1,-1), 2)
                    response.append("")
                    response.append(transcribe_streaming(single_speaker, client))
                # Process the aggregated audio data
                # audio_array = torch.frombuffer(audio_data, dtype=torch.float32)
                # to_save = np.frombuffer(audio_data, dtype=np.float32).copy()
                

                #sf.write('test_save{}.wav'.format(j),to_save, FS, subtype='FLOAT')
                #  j += 1
                audio_data.clear()
                speaker1_split = None
                speaker2_split = None
                # speaker1_split, speaker2_split = matcher.assign_speaker(audio_array)
                # audio_array = speaker1_split.cpu().numpy()
                # audio_array2 = speaker2_split.cpu().numpy()
                #matcher.addSampleBatch(audio_array.view(1,-1), 1)
               
                #response = []
                #response.append(transcribe_streaming(to_save, client))
                #response.append(transcribe_streaming(audio_array2, client))
                msg_queue.put((response, time.time()))
                print("Buffered audio processed.")
            else:
                print("No audio to process.")


def transcribe_streaming(audio_data: np.ndarray,client) -> speech.RecognitionConfig:
    """Streams transcription of the given audio file."""
    # if audio_data.dtype != np.int16:
    #     audio_data = (audio_data * 32767).astype(np.int16)  # Convert float to int16
    # audio_bytes = audio_data.tobytes()
    # In practice, stream should be a generator yielding chunks of audio data.
    
    if audio_data.dtype != np.int16:
        audio_data = (audio_data * 32767).astype(np.int16)  # Convert float to int16
    audio_bytes = audio_data.tobytes()
    # In practice, stream should be a generator yielding chunks of audio data.
    stream = [audio_bytes]
    requests = (
        speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in stream
    )
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=FS,
        language_code="en-US",
    )
    
    streaming_config = speech.StreamingRecognitionConfig(config=config)
    # streaming_recognize returns a generator.
    responses = client.streaming_recognize(
        config=streaming_config,
        requests=requests,
    )
    for response in responses:
        # Once the transcription has settled, the first result will contain the
        # is_final result. The other results will be for subsequent portions of
        # the audio.
        for result in response.results:
            print(f"Finished: {result.is_final}")
            print(f"Stability: {result.stability}")
            alternatives = result.alternatives
            # The alternatives are ordered from most likely to least.
            for alternative in alternatives:
                with open("example.txt", 'a') as file:
                        # Write new text to the file
                        file.write("separation result:")
                        file.write(alternative.transcript)
                        file.write("\n")
                return alternative.transcript
    return ""






        
def main():
    global flag_run_ctrl_c
    flag_run_ctrl_c=True
    signal.signal(signal.SIGINT, handler_SIGINT)
    paras={}
    paras['cam_idx']=0

    paras['FORMAT']             =FORMAT  
    paras['n_channels']         =CHANNELS
    paras['fs']                 =FS
    paras['n_samples_per_chunk']=CHUNK
    paras['frame_audio_shape']=(paras['n_channels'], paras['n_samples_per_chunk'])   # (# of channels, # of samples)

    #frame_audio_share = multiprocessing.Array(ctypes.c_int32, paras['frame_audio_shape'][0]*paras['frame_audio_shape'][1])
    audio_queue = multiprocessing.Queue()
    silence_detected_event = multiprocessing.Event()
    message_queue = multiprocessing.Queue()
    stream_queue = multiprocessing.Queue()
    flag_run =multiprocessing.Value('i', 1)
    speaker1_talking = multiprocessing.Value(ctypes.c_bool, False)
    speaker2_talking = multiprocessing.Value(ctypes.c_bool, False)
    #p_cam    =multiprocessing.Process(target=loop_cam,      args=(flag_run, paras,))
    p_mic    =multiprocessing.Process(target=loop_mic_v2,   args=(flag_run, audio_queue, silence_detected_event, paras))
    p_stream = multiprocessing.Process(target=single_mic, args=(stream_queue,))
    p_ml    =multiprocessing.Process(target=loop_ML,   args=(flag_run, audio_queue, silence_detected_event,message_queue,speaker1_talking,speaker2_talking,paras))
    


    #p_cam.start()
    p_mic.start()
    p_ml.start()
    p_stream.start()

    print("Waiting for mic and cam to setup...")
    time.sleep(2)
    flag_run.value = 1


    PREDICTOR_PATH = r"face_detector/shape_predictor_68_face_landmarks.dat"
    LIP_DIST_CUTOFF = 10.0

    detector = dlib.get_frontal_face_detector()

    cap = cv2.VideoCapture(0)
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    last_message = None
    stream_message = None
    message_display_time = 0
    display_duration = 5

    while True:
        start = time.time()

        success, image = cap.read()
        img_h, img_w = image.shape[:2]
        # detector usage if using dnn
        # blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        # net.setInput(blob)
        # detections = net.forward()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        speaker1_talking.value = False
        speaker2_talking.value = False
        index = 0
        # draw bounding box
        index = 0
        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            landmarks = np.matrix([[p.x, p.y]  
                for p in predictor(image, dlib_rect).parts()])
            mouth_point_up = landmarks[62]
            mouth_point_down = landmarks[66]
            # print out landmark for upper lip and lower lip
            counter = 0
            for point in (mouth_point_up, mouth_point_down):  
                pos = (point[0, 0], point[0, 1])
                cv2.circle(image, pos, 2, color=(0, 255, 255), thickness=-1)
                counter+= 1
            dist = np.linalg.norm(mouth_point_up-mouth_point_down)
            #delayed display of captions
            
            if index == 0:
            # if it's the first face, we set speaker num based on if face is on left or right
                if x + w / 2 < img_w / 2:
                    speaker_num = 0
                else:
                    speaker_num = 1
            else:
            # if second face, we just switch speaker num
                speaker_num = 1-speaker_num
            current_time = time.time()
            if dist > LIP_DIST_CUTOFF or (last_message and current_time - message_display_time < display_duration):
                if current_time - message_display_time >= display_duration:
                    try:
                        last_message, message_display_time = message_queue.get_nowait()
                    except queue.Empty:
                        pass  # No new message, keep displaying the last one

                if last_message:
                    text = last_message[speaker_num]
                    font = 1
                    fontScale = 2
                    thickness = 2
                    text_size, _ = cv2.getTextSize(text, font, fontScale, thickness)
                    text_w, text_h = text_size
                    cv2.rectangle(image, (x, y - 10 - text_h), (x + text_w, y), (0, 0, 0), -1)
                    cv2.putText(image, text, (x, y - 10), 
                                fontFace=1,
                        fontScale=2,thickness = 2,
                        color=(0, 255, 255))
                if speaker_num == 0:
                    speaker1_talking.value = True
                else:  
                    speaker2_talking.value = True
                
            else:
                # Reset last_message when not speaking or after duration has passed
                last_message = None
                if speaker_num == 0:
                    speaker1_talking.value = False
                else:  
                    speaker2_talking.value = False
            index += 1

            current_time = time.time()
            stream_display_time = 0
            if current_time - stream_display_time >= display_duration:
                try:
                    stream_message, stream_display_time = stream_queue.get_nowait()
                        
                except queue.Empty:
                    pass
            if stream_message:
                #print("print stream message"+stream_message)
                #print("code reached")
                # if stream_message != "":
                example_bottom_string = stream_message
                bottom_font = 1
                bottom_fontScale = 1.5
                bottom_thickness = 2
                text_size, _ = cv2.getTextSize(example_bottom_string, bottom_font, bottom_fontScale, bottom_thickness)
                text_w, text_h = text_size
                bottom_start = (img_w // 2 - text_w // 2, img_h - text_h - 10)
                bottom_end = (img_w // 2 + text_w // 2, img_h - 10)
                cv2.rectangle(image, bottom_start, bottom_end, (0, 0, 0), -1)
                cv2.putText(image, example_bottom_string, (bottom_start[0], bottom_start[1] + text_h),  
                        fontFace= bottom_font,  
                        fontScale= bottom_fontScale,
                        thickness = bottom_thickness,
                        color=(0, 255, 255))
                


        cv2.imshow("Webcam", image) # This will open an independent window
        if cv2.waitKey(1) & 0xFF==ord('q'): # quit when 'q' is pressed
            cap.release()
            break

    message_queue.close()
    message_queue.join_thread()

    cv2.destroyAllWindows() 
    cv2.waitKey(1)
    
    p_mic.join()
    p_ml.join()
    print(" stopped properly ~")
    sys.exit()
                


    


if __name__ == "__main__":
    # Start the background thread
    # message_queue = queue.Queue()
    # bg_thread = threading.Thread(target=background_task, args=(message_queue,))
    # bg_thread.daemon = True  # Daemonize thread
    # bg_thread.start()
    # main_thread = threading.Thread(target=main, args=(message_queue,))
    # main_thread.daemon = True
    # main_thread.start()

    # try:
    #     def important(x):
    #         return chr(x)
    #     i = "I"
    #     a = important(ord("A")-10 + 8 + 2)
    #     o = important("O")
    #     x = important(ord("X"))
    #     last_name = f"{x}{i}{a}{o}"
    #     name = f"{last_name} Huo"
    #     message = f"Happy Birthday {name}!"
    #     print(f"{message}")
    # except Exception as e:
    #     print("Happy Birthday Xiao Huo!")
    main()

    
    # main()