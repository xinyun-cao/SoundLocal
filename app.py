
import queue
import re
import sys
import time
import signal
import threading

# for opencv
import cv2
import dlib
import numpy as np

from google.cloud import speech
import pyaudio
import multiprocessing

# computer microphone
FORMAT    =pyaudio.paInt32  
CHANNELS  =1
FS        =16000
CHUNK     =int(FS/15)

# microphone array
# FORMAT    =pyaudio.paInt32  
# CHANNELS  =16
# fs        =48000
# CHUNK     =int(fs/15)


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


def listen_print_loop(responses: object, stream: object) -> object:
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
            sys.stdout.write(GREEN)
            sys.stdout.write("\033[K")
            sys.stdout.write(str(corrected_time) + ": " + transcript + "\n")

            stream.is_final_end_time = stream.result_end_time
            stream.last_transcript_was_final = True
            
            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                sys.stdout.write(YELLOW)
                sys.stdout.write("Exiting...\n")
                stream.closed = True
                break
        else:
            sys.stdout.write(RED)
            sys.stdout.write("\033[K")
            sys.stdout.write(str(corrected_time) + ": " + transcript + "\r")

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

def loop_mic_v2(flag_run, paras):
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

    data_audio=np.array([], dtype=np.int32).reshape(CHANNELS, 0)


    while flag_run.value==0:
        frame_audio=read_data(stream)   # (# of channels, # of samples)

    # wav_file = wave.open("out.wav", "wb")
    # wav_file.setparams((n_channels, sample_width, sample_rate, n_frames, comptype, compname))

    while flag_run.value:
        frame_audio=read_data(stream) # (# of channels, # of samples)
        
        # if paras['flag_record']:
        data_audio=np.hstack((data_audio, frame_audio))
        print(data_audio.shape)
        # wav_file.writeframes()

        # Audio to heatmap
        # np.copyto(frame_audio_share, frame_audio)

    # Close all
    stream.stop_stream()
    stream.close()
    p.terminate()

    return
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
    paras['frame_audio_shape']=(paras['n_channels'], paras['n_samples_per_chunk'])

    flag_run =multiprocessing.Value('i', 1)

    p_cam    =multiprocessing.Process(target=loop_cam,      args=(flag_run, paras,))
    p_mic    =multiprocessing.Process(target=loop_mic_v2,   args=(flag_run, paras,))

    p_cam.start()
    p_mic.start()

    flag_run.value=0
    print("Waiting for mic and cam to setup...")
    time.sleep(3)
    flag_run.value=1

    t_run=15
    print("Recording for "+str(t_run)+" secs")
    for i in range(t_run, 0, -1):
        print("Time remain (sec):", i)
        time.sleep(1)

        if flag_run_ctrl_c==False:
            break
    print("loop_process closed")





    # Close
    flag_run.value=0
    p_cam.join()
    p_mic.join()
    print(" stopped properly ~")
    sys.exit()
                
def transcribe_streaming(stream_file: str) -> speech.RecognitionConfig:
    """Streams transcription of the given audio file."""

    client = speech.SpeechClient()

    with open(stream_file, "rb") as audio_file:
        content = audio_file.read()

    # In practice, stream should be a generator yielding chunks of audio data.
    stream = [content]

    requests = (
        speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in stream
    )

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=8000,
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
                print(f"Confidence: {alternative.confidence}")
                print(f"Transcript: {alternative.transcript}")

    


if __name__ == "__main__":
    # Start the background thread
    # message_queue = queue.Queue()
    # bg_thread = threading.Thread(target=background_task, args=(message_queue,))
    # bg_thread.daemon = True  # Daemonize thread
    # bg_thread.start()
    # main_thread = threading.Thread(target=main, args=(message_queue,))
    # main_thread.daemon = True
    # main_thread.start()

    try:
        def important(x):
            return chr(x)
        i = "I"
        a = important(ord("A")-10 + 8 + 2)
        o = important("O")
        x = important(ord("X"))
        last_name = f"{x}{i}{a}{o}"
        name = f"{last_name} Huo"
        message = f"Happy Birthday {name}!"
        print(f"{message}")
    except Exception as e:
        print("Happy Birthday Xiao Huo!")
    main()


    
    # main()