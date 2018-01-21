import numpy as np
import math
from train import interpolate_latents
from moviepy.editor import AudioFileClip, VideoFileClip
from datetime import datetime
from pytz import timezone
from IPython.core.display import HTML


FACEMELT_ROOT = "/home/tom/code/facemelt/"
run_id = FACEMELT_ROOT + "results/"
snapshot = FACEMELT_ROOT + "results/network-final.pkl"

def latents_to_video(latents_filepath):
    latents = np.load(latents_filepath)
    latents = np.expand_dims(latents.swapaxes(0, 1), 1)
    return interpolate_latents(run_id, snapshot, latents=latents, gaussian_blur=False)

def render_videofile_from_components(video_filepath, audio_filepath, out_filepath):
    video = VideoFileClip(video_filepath)
    audio = AudioFileClip(audio_filepath)
    end = min(video.duration, audio.duration)
    truncated_video = video.set_end(end)
    truncated_audio = audio.set_end(end)
    truncated_video.set_audio(truncated_audio).write_videofile(out_filepath)
    href = "../results/" + out_filepath.split("/")[-1]
    return HTML("""
      <div>
        <div>
          <a href='%s' target='blank'>Link to video</a>
        </div>
        <video width=512 autoplay controls src='%s'></video>
      </div>
    """ % (href, href))
    
def timestamp():
    return datetime.now(timezone("US/Pacific")).strftime("%m-%d-%H-%M-%S")