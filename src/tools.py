import os
import tempfile
import shlex
from bs4 import BeautifulSoup
import inspect
import requests
from typing import Any
import base64

import aisuite
import subprocess
import duckduckgo_search as ddgs
from src.util import extract_text, format_subproc_result, sanitize_path

MAX_RESPONSE_LENGTH = 1000000

# Fetching the OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BING_SEARCH_API_KEY = os.getenv("BING_SEARCH_API_KEY")


# This is the logging utility used to log from tools
def log_tool(**kwargs: Any):
    # Get the name of the calling function from the call stack
    frame = inspect.currentframe()
    caller_frame = frame.f_back if frame is not None else None

    name = (
        caller_frame.f_code.co_name
        if caller_frame is not None
        else '<unknown>'
    )

    # Format each key-value pair in kwargs as key: [value]
    args_str = ", ".join(f"{key}: [{value}]" for key, value in kwargs.items())
    print(f"[{name}]" + (", " + args_str if args_str else ""))

    def p(string: str):
        print("    " + string)

    return p


def run_shell_command(cmd: str, timeout: int = 30):
    """
    This tool runs a shell command on the user's machine and is expected to
    fulfill about most of the user's requests. However, other tools are
    still available and should be used as needed.

    Note: A 124 status code indicates a timeout. Try increasing timeout and
    running again.

    Args:
        cmd: The shell command string to run.
        timeout: The maximum time in seconds to allow the command to run
                (default is 30).

    Returns:
        The formatted output of the shell command execution,
        including stdin, stdout, and stderr.

    """

    p = log_tool(cmd=cmd, timeout=timeout)

    # Long running commands will hose the agent, so let's prevent that:
    cmd = f"timeout {timeout} {cmd}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    p(f"⮑  {result.returncode} " + (result.stderr.splitlines()[0] if result.stderr else ""))
    if result.stdout:
        ellipses = ""
        if len(result.stdout.splitlines()) > 1:
            ellipses = "..."

        first_line = result.stdout.splitlines()[0]

        p(f"⮑  {first_line}{ellipses}")

    p("")
    return format_subproc_result(result)


def fetch(url: str):
    """Fetch content from the provided URL."""

    p = log_tool(url=url)

    response = requests.get(url, allow_redirects=True)
    try:
        response.raise_for_status()
        text = extract_text(response.text)

        if len(text) > MAX_RESPONSE_LENGTH:
            p("truncating response")
            return (
                f"{text[:MAX_RESPONSE_LENGTH]}\n\n"
                "[Content truncated due to size limitations]"
            )

        p(f"[{str(text.count('\n') + 1)}] lines fetched")
        return f"[URL]: {url}\n\n" + text

    except Exception as e:
        p(f"❌ status: [{response.status_code}], e: [{e}]")
        return f"Error fetching URL {url}: {e}"


def search_text(text: str, max_results: int = 3):
    """
    Search the web for the provided text
    Limit the search to max_results
    """

    p = log_tool(text=text, max_results=max_results)

    dds = ddgs.duckduckgo_search.DDGS()
    return_str = ""
    results = []

    try:
        results = dds.text(text, max_results=max_results, backend="lite")
    except Exception as e:
        p(f"X duck duck go search error: [{e}]")
        return_str += f"duck duck go search api error: [{e}]\n"

    for i, result in enumerate(results, start=1):
        p(f"-> {result['href']}")
        return_str += f"{i}. {result['title']}\n   {result['href']}\n"

    return return_str


def search_text_alternative(text: str, max_results: int = 3):
    """
    Alternative web search for when the primary one fails.
    """

    p = log_tool(text=text, max_results=max_results)

    # Using curl instead of requests to avoid bot detection
    query = shlex.quote(text)

    curl_command = (
        f"curl -A 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) "
        "Gecko/20100101 Firefox/117.0' "
        f"-s https://html.duckduckgo.com/html/?q={query}"
    )

    result = subprocess.run(
        curl_command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=10
    )

    if result.returncode != 0:
        p(f"❌ curl command code [{result.returncode}]: {result.stderr}")
        return "DuckDuckGo search error:\n" + format_subproc_result(result)

    soup = BeautifulSoup(result.stdout, "html.parser")

    return soup.text


def search_images(text: str, max_results: int = 3):
    """
    Search the web for images that match the provided text
    Limit the search to max_results
    """

    p = log_tool(text=text, max_results=max_results)

    dds = ddgs.duckduckgo_search.DDGS()
    results = dds.images(text, max_results=max_results)

    text = ""

    for i, result in enumerate(results or [], start=1):
        p(f" {result['image']}")
        text += f"{i}. {result['title']}\n   {result['image']}\n"

    return text


def printz(cmd: str):
    """
    Place a command on the user's command buffer.
    Use this when the user has asked for a command to do such and such.
    Named after the zsh print -z command.
    """

    log_tool(cmd=cmd)


def gen_image(
    number: int,
    model: str,
    size: str,
    prompt: str
):
    """
    Generate an image

    Use gen_image only when explicitly asked for an image,
    like 'generate an image of ..', or 'make a high quality image of ..'.
    If the user has asked for a generated or created image, use this method.

    Arguments:
        number: number of images to create in parallel
        model: either "dall-e-2" for low to normal quality images,
               or "dall-e-3" for high quality images
        size: the image dimensions
        prompt: what text to give to the AI image creation service
    """

    p = log_tool(number=number, model=model, size=size, prompt=prompt)

    url = "https://api.openai.com/v1/images/generations"
    data = {
        "n": number,
        "model": model,
        "size": size,
        "prompt": prompt
    }

    response = requests.post(
        url,
        json=data,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
    )

    try:
        response.raise_for_status()
        return response.text

    except Exception as e:
        p(f"❌ status: [{response.status_code}]")
        return f"Error creating images: {e}"


def read_file(path: str):
    """
    Read a file from the file system

    Args:
        path - relative path, ex. ./file.ext, or absolute path.
            Must contain folder, even if just ./
    """

    path = sanitize_path(path)

    p = log_tool(path=path)

    try:
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        p(f"❌ File not found: {path}")
        return f"file not found: {path}"
    except IOError as e:
        p(f"❌ IOError while reading file: {e}")
        return f"IOError while reading file: {e}"


def write_file(path: str, contents: str):
    """
    Write {contents} to the file at {path}
    Overwrites file

    Args:
        path - relative path, ex. ./file.ext, or absolute path.
            Must contain folder, even if just ./

        contents - the contents you want written to the file
    """

    path = sanitize_path(path)

    # Give a line count
    p = log_tool(path=path, contents=f"{str(contents.count('\n') + 1)} lines")

    try:
        # Ensure the directory exists, create if not
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception as e:
        p(f"❌ Error creating directories: {e}")
        return f"Error creating directories: {e}"

    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(contents)
    except Exception as e:
        p(f"❌ Error writing to file: {e}")
        return f"Error writing to file: {e}"

    return "Success"


def patch_file(path: str, diff: str):
    """
    Apply a unified diff string patch to the file at file_path.

    Args:
        path: path to the target file to patch
        diff: the unified diff string to apply

    **How to ensure reliable patching:**
    - Always use proper unified diff format as produced by a command like:
      `diff -u oldfile.py newfile.py > file.patch`
    - Confirm hunk headers look like: `@@ -7,7 +7,16 @@` (not function names).

    Returns:
        A string indicating success or error message
    """

    file_path = sanitize_path(path)
    p = log_tool(path=path, diff=f"{str(diff.count('\n') + 1)} lines")

    # Write the diff content to a temporary patch file
    tmp_patch_file_path = file_path + ".patch_temp"
    try:
        with open(tmp_patch_file_path, "w", encoding="utf-8") as patch_file:
            patch_file.write(diff)
    except Exception as e:
        p(f"❌ Error writing patch file: {e}")
        return f"Error writing patch file: {e}"

    # Patch expects the header to match the target file; patch works from the same directory
    # So use only the filename in the diff header, and run patch in the file's directory.
    cwd = os.path.dirname(file_path) or "."
    filename = os.path.basename(file_path)

    # Pre-check: Ensure the diff header matches the basename
    # Optionally rewrite the diff if necessary:
    first_lines = diff.splitlines()
    if len(first_lines) >= 2 and (
        first_lines[0].startswith('--- ') and first_lines[1].startswith('+++ ')
    ):
        def fix_header(line, newfile):  # patch header rewrite helper
            parts = line.split()
            if len(parts) > 1:
                return f"{parts[0]} {newfile}"
            return line
        fixed_diff = (
            fix_header(first_lines[0], filename) + "\n" +
            fix_header(first_lines[1], filename) + "\n" +
            "\n".join(first_lines[2:])
        )
        try:
            with open(tmp_patch_file_path, "w", encoding="utf-8") as patch_file:
                patch_file.write(fixed_diff)
        except Exception as e:
            p(f"❌ Error rewriting patch file: {e}")
            return f"Error rewriting patch file: {e}"

    # Apply the patch
    try:
        cmd = f"patch -u {shlex.quote(filename)} -i {shlex.quote(os.path.basename(tmp_patch_file_path))}"
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=cwd
        )

        if result.stdout:
            p("[patch stdout] " + result.stdout.strip())
        if result.stderr:
            p("[patch stderr] " + result.stderr.strip())

        if result.returncode != 0:
            # Return all relevant output for debugging
            return (
                f"Patch command failed:\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

    finally:
        try:
            os.remove(tmp_patch_file_path)
        except Exception as e:
            p(f"Warning: could not remove temporary patch file: {e}")

    return "Patch applied successfully"

def build_trim_message(messages: list[aisuite.Message]):
    def trim_message(index: int, new_content: str):
        """
        Remove or rewrite one of the messages in the history of this chat.

        Args:
            index: the position in the messages array of the message to modify
            new_content:
                If empty string, message is deleted.
                Otherwise, message content is overwritten with new_content
        """

        p = log_tool(index=index, new_content=new_content)

        try:
            if new_content == "":
                del messages[index]
            else:
                messages[index].content = new_content

            p(f" message {index} trimmed")
            return f"message[{index}] successfully modified"
        except Exception as e:
            p(f" trim failed: {e}")
            return f"tool trim_message failed: {e}"

    return trim_message


def summarize_response(summary: str):
    """
    [MANDATORY]
    This is one of the primary ways the user communicates with you.
    After every response, you should speak aloud a very quick summary
    of that response. If the user then asks for clarification, you can
    speak for longer.

    This tool uses the macOS native 'say' command for speech synthesis.
    The summary should be brief, generally taking no more than 5 seconds
    to read aloud.
    Only extend the spoken summary length if the user explicitly requests
    further clarification or details.

    Args:
        summary: The very short summary string to be spoken aloud
                 in under 5 seconds.
    """
    p = log_tool(summary=summary)

    # Sanitize summary input to avoid injection (simple quote escape)
    safe_summary = summary.replace('"', '\"')

    # Use subprocess to call say asynchronously
    try:
        subprocess.Popen(['say', safe_summary])
        p("Speech synthesis started")
        return "Speech synthesis started"
    except Exception as e:
        p(f"Error starting speech synthesis: {e}")
        return f"Error starting speech synthesis: {e}"


def spawn(task: str):
    """
    Spawn a new single-invocation AI agent to complete a specific task.

    Creates a fresh agent using aisuite that can work independently and
    return focused results. The spawned agent has access to basic tools.

    Args:
        task: The specific task description for the spawned agent to complete

    Returns:
        The output from the spawned agent's completion of the task
    """
    p = log_tool(task=task)

    try:
        # Create a lightweight aisuite client for single-shot execution
        import aisuite as ai

        client = ai.Client()

        # Create a focused system prompt for the spawned agent
        spawn_system = (
            "You are a single-purpose AI agent spawned to complete one specific task. "
            "Focus solely on the given task, be concise, and provide actionable results. "
            "Complete the task efficiently and return your findings."
        )

        messages = [
            ai.Message(role="system", content=spawn_system),
            ai.Message(role="user", content=task)
        ]

        # Use a fast model with basic tools for the spawned agent
        response = client.chat.completions.create(
            model="openai:gpt-4o-mini",
            messages=[message.model_dump() for message in messages],
            tools=[
                fetch,
                run_shell_command,
                read_file,
            ],
            max_turns=10  # Limit turns for focused execution
        )

        choice = response.choices[0]
        result = choice.message.content

        p(f"spawned agent completed, returned {len(result)} characters")
        return f"[SPAWNED AGENT OUTPUT]\n{result}"

    except Exception as e:
        p(f"❌ spawn failed: {e}")
        return f"Error spawning agent: {e}"


# This is great probably, but doesn't work due to langchain limitations
# and weird provider rules regarding image uploads:
# https://github.com/langchain-ai/langchain/discussions/25881
def screenshot_and_upload(area: str = "screen"):
    """
    Captures a screenshot and returns a base64-encoded string of the image.

    Args:
        area: 'screen' (default, full screen), or a region in format 'x,y,w,h'.

    Returns:
        Dict with {'base64': ..., 'message': ...}.
    """
    p = log_tool(area=area)
    tmpfile = tempfile.mktemp(suffix='.png')

    # Build screencapture command
    cmd = ["screencapture", "-x"]
    if area != "screen":
        coords = area.split(",")
        if len(coords) == 4:
            rect_arg = "-R{},{},{},{}".format(*coords)
            cmd.append(rect_arg)
    cmd.append(tmpfile)

    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        return {"base64": None, "message": f"Screenshot failed: {e}"}

    try:
        with open(tmpfile, "rb") as image_file:
            img_bytes = image_file.read()
            base64_str = base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        return {
            "base64": None,
            "message": f"Failed to read screenshot file: {e}"
        }
    finally:
        try:
            os.remove(tmpfile)
        except Exception:
            pass

    b64_shortened = f"{base64_str[:5]}...{base64_str[-5:]}"
    p(f"screenshot captured and base64 encoded: [{b64_shortened}]")

    return [{
        "type": "image",
        # "source": { # This would be anthropic's style
        #     "type": "base64",
        #     "mime_type": "image/png",  # or image/png, etc.
        #     "data": base64_str,
        # },
        "source_type": "base64",
        "mime_type": "image/png",  # or image/png, etc.
        "data": base64_str,
    }]
