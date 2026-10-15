"""Describing a clip with Gemini: the prompt, the structure of the answer, and the call."""
import subprocess
import time
from typing import List

from pydantic import BaseModel, Field

# The rules keep the descriptions to what is on screen. Labels such as "migrant" are left out
# on purpose: the content space should reflect what is shown, not how it is framed.
PROMPT = """Describe what is VISIBLE in this silent video clip, in English.
Be exhaustive: more detail is better than less. Mention numbers, colours,
positions and any readable text.

Rules:
- Describe only what can be seen. Do not interpret the meaning of the clip.
- Never use the words migrant, refugee or asylum seeker.
- Do not state or guess ethnicity, nationality or religion.
- Do not give timestamps.
- Transcribe text exactly as it appears. If text cannot be read with certainty,
  write "unreadable text" and say where it is; do not guess."""


# The answer must follow this structure. The field descriptions are sent to the model as
# instructions, so they are part of the prompt.
class Shot(BaseModel):
    description: str = Field(description="One or two sentences: where this shot is, who is in it, "
                                         "what happens, and how it is framed.")


class ClipDescription(BaseModel):
    summary: str = Field(description="Two or three sentences describing the whole clip.")
    shots: List[Shot] = Field(description="Every shot in order; a new shot starts at every cut.")
    setting: str = Field(description="Place type, indoor/outdoor, time of day, weather, landscape.")
    people: str = Field(description="How many people, apparent age groups, clothing, uniforms.")
    actions: str = Field(description="What people do, in order.")
    objects: str = Field(description="Vehicles, boats, tents, fences, belongings, other objects.")
    overlay_text: str = Field(description="Text and graphics added by the broadcaster: channel logos, "
                                          "captions, subtitles, name bars. Say where on screen. Empty if none.")
    scene_text: str = Field(description="Text that is part of the filmed scene: signs, vehicles, "
                                        "clothing, banners. Empty if none.")
    camera: str = Field(description="Shot types, camera movement, aerial or handheld.")
    uncertain: str = Field(description="Anything that could not be made out clearly. Empty if none.")


def describe_clip(client, video_id, video_dir, model, sample_fps):
    """Send a muted copy of one clip to Gemini and return its description as a dict."""
    # Muted, so the description rests on the picture alone and not on the voice-over.
    src   = video_dir / f"{video_id}.mp4"
    muted = video_dir / f"{video_id}.muted.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-an", "-c:v", "copy", str(muted)],
                   check=True)

    # Upload, wait until Gemini has processed the file, then delete the temporary muted copy.
    upload = client.files.upload(file=str(muted))
    while not upload.state or upload.state.name != "ACTIVE":
        time.sleep(2)
        upload = client.files.get(name=upload.name)
    muted.unlink()

    interaction = client.interactions.create(
        model=model,
        input=[
            {"type": "video", "uri": upload.uri, "mime_type": upload.mime_type,
             "processing": {"type": "static", "fps": sample_fps}},
            {"type": "text", "text": PROMPT},
        ],
        response_format={"type": "text", "mime_type": "application/json",
                         "schema": ClipDescription.model_json_schema()},
    )
    result = ClipDescription.model_validate_json(interaction.output_text).model_dump()
    # Store the shots as one numbered text ("1. ...", "2. ..."), which load_shots splits again.
    result["n_shots_described"] = len(result["shots"])
    result["shots"] = "\n".join(f"{i}. {s['description']}"
                                for i, s in enumerate(result["shots"], start=1))
    return result
