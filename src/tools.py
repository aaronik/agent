import os
import shlex
from bs4 import BeautifulSoup
import inspect
import requests
from typing import Any

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
    The primary tool you should use to accomplish the task.

    This tool runs a shell command on the user's machine and is expected to
    fulfill about 90% of the user's task requests. However, other tools are
    still available and should be used as needed for specialized tasks.

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

    p(f"⮑  {result.returncode}")

    return format_subproc_result(result)


def fetch(url: str):
    """Fetch content from the provided URL."""

    p = log_tool(url=url)

    response = requests.get(url, allow_redirects=True)
    try:
        response.raise_for_status()
        text = extract_text(response.text)

        if len(text) > MAX_RESPONSE_LENGTH:
            p(" truncating response")
            return (
                f"{text[:MAX_RESPONSE_LENGTH]}\n\n"
                "[Content truncated due to size limitations]"
            )

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
        p(f" duck duck go search error: [{e}]")
        return_str += f"duck duck go search api error: [{e}]\n"

    for i, result in enumerate(results, start=1):
        p(f" {result['href']}")
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


def apply_diff(path: str, diff: str):
    """
    Apply a unified diff string patch to the file at file_path.

    Args:
        path: path to the target file to patch
        diff: the unified diff string to apply

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

    # Apply the patch
    try:
        # Flags:
        # -u unified diff format
        # -r disables reject files generation
        cmd = f"patch -u {file_path} -i {tmp_patch_file_path} -r -"
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            p(f"❌ Patch command failed: {result.stderr.strip()}")
            return f"Patch command failed: {result.stderr.strip()}"
    finally:
        try:
            os.remove(tmp_patch_file_path)
        except Exception as e:
            p(f" Warning: could not remove temporary patch file: {e}")

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

    Speak a ~5 second summary using system speech synthesizer
    (macOS 'say' command). It should take no more than 5 seconds
    to read back the summary.

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
