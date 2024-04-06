
import queue
import re
import sys
import time
from asteroid.models import ConvTasNet
import threading
# for opencv
import cv2
import dlib
import numpy as np
import samplerate as sr
from google.cloud import speech
import pyaudio
import multiprocessing

# Audio recording parameters
STREAMING_LIMIT = 240000  # 4 minutes
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms
CHUNK_SIZE = SAMPLE_RATE *3 # 3 seconds

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"


def get_current_time() -> int:
    """Return Current Time in MS.

    Returns:
        int: Current Time in MS.
    """

    return int(round(time.time() * 1000))


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
            
def bytes_to_floats(byte_data):
    # Convert byte data to numpy array of type int16 (for 16-bit PCM)
    int_data = np.frombuffer(byte_data, dtype=np.int16)
    
    # Convert int16 data to float (ranging from -1.0 to 1.0)
    float_data = int_data.astype(np.float32) / 32768.0
    
    return float_data

def floats_to_bytes(float_data):
    # Ensure the input is a float32 numpy array
    if not isinstance(float_data, np.ndarray) or float_data.dtype != np.float32:
        float_data = np.array(float_data, dtype=np.float32)
    
    # Scale the float data back to int16 range
    int_data = (float_data * 32767).astype(np.int16)
    
    # Convert the int16 data to bytes
    byte_data = int_data.tobytes()
    
    return byte_data

def normalize(audio):
        return audio / np.max(np.abs(audio))

def resample(audio, sr_old, sr_new):
        if sr_old == sr_new:
            return audio
        else:
            resampler = sr.Resampler()

            resampled_len = len(resampler.process(audio[0], sr_new/sr_old))
            resampled_len = len(resampler.process(audio[0], sr_new/sr_old))
            # I can't explain it but it's wrong unless run twice
            resampled_audio_mixed = np.zeros((audio.shape[0], resampled_len))
            for i in range(audio.shape[0]):
                resampled_audio_mixed[i] = normalize(resampler.process(audio[i], sr_new/sr_old))[:resampled_len]
            return resampled_audio_mixed
        
def transcribe_streaming(audio_data: np.ndarray) -> speech.RecognitionConfig:
    """Streams transcription of the given audio file."""

    client = speech.SpeechClient()

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
            return result.is_final

                 
def main(msg_queue):
    """start bidirectional streaming from microphone input to speech API"""
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="en-US",
        max_alternatives=1,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    mic_manager = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)
    sys.stdout.write(YELLOW)
    sys.stdout.write('\nListening, say "Quit" or "Exit" to stop.\n\n')
    sys.stdout.write("End (ms)       Transcript Results/Status\n")
    sys.stdout.write("=====================================================\n")
    model = ConvTasNet.from_pretrained("mpariente/ConvTasNet_Libri3Mix_sepnoisy")
    with mic_manager as stream:
        while not stream.closed:
            sys.stdout.write(YELLOW)
            sys.stdout.write(
                "\n" + str(STREAMING_LIMIT * stream.restart_counter) + ": NEW REQUEST\n"
            )

            stream.audio_input = []
            audio_generator = stream.generator()
            for content in audio_generator:
                audio_mixed = resample(bytes_to_floats(audio_generator), 16000, 8000)
                print(audio_mixed.shape)
            # separated = model.separate(audio_mixed)

            # responses = transcribe_streaming(separated[0][0])

                # requests = (
                #     speech.StreamingRecognizeRequest(audio_content=content)
                #     for content in audio_generator
                # )

                # responses = client.streaming_recognize(streaming_config, requests)

                # # Now, put the transcription responses to use.
                # response = listen_print_loop(responses, stream)
                # if stream.result_end_time > 0:
                #     stream.final_request_end_time = stream.is_final_end_time
                # stream.result_end_time = 0
                # stream.last_audio_input = []
                # stream.last_audio_input = stream.audio_input
                # stream.audio_input = []
                # stream.restart_counter = stream.restart_counter + 1

                # msg_queue.put((response, time.time()))

                # if not stream.last_transcript_was_final:
                #     stream.new_stream = True
                #     sys.stdout.write("\n")
                
            


if __name__ == "__main__":
    # Start the background thread
    # message_queue = queue.Queue()
    # bg_thread = threading.Thread(target=background_task, args=(message_queue,))
    # bg_thread.daemon = True  # Daemonize thread
    # bg_thread.start()
    # main_thread = threading.Thread(target=main, args=(message_queue,))
    # main_thread.daemon = True
    # main_thread.start()

    message_queue = multiprocessing.Queue()
    bg_process = multiprocessing.Process(target=main, args=(message_queue,))
    bg_process.daemon = True
    bg_process.start()


    PREDICTOR_PATH = r"face_detector/shape_predictor_68_face_landmarks.dat"
    LIP_DIST_CUTOFF = 5.0

    detector = dlib.get_frontal_face_detector()

    cap = cv2.VideoCapture(0)
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    last_message = None
    message_display_time = 0
    display_duration = 3

    while True:
        start = time.time()
        
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
                        last_message, message_display_time = message_queue.get_nowait()
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
    
    message_queue.close()
    message_queue.join_thread()
    bg_process.terminate()
    bg_process.join()
    cv2.destroyAllWindows() 
    cv2.waitKey(1)
    # main()