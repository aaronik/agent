import os
import tempfile
import shlex
from bs4 import BeautifulSoup
import requests
from typing import Any
import base64

import aisuite
import subprocess
import duckduckgo_search as ddgs
from src.util import format_subproc_result, sanitize_path
from src.tool_status_display import get_tool_status_display

MAX_RESPONSE_LENGTH = 1000000

# Fetching the OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BING_SEARCH_API_KEY = os.getenv("BING_SEARCH_API_KEY")


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

    # Long running commands will hose the agent, so let's prevent that:
    cmd = f"timeout {timeout} {cmd}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return format_subproc_result(result)


def fetch(url: str):
    """Fetch content from the provided URL using jina.ai reader.

    This function intentionally avoids double-prefixing URLs that already
    target the jina.ai reader. If the caller passes a URL that already
    starts with "https://r.jina.ai/", we use it as-is. Otherwise, we
    prefix the provided URL with the reader base.
    """

    # Avoid double-prefixing when the caller already passed an r.jina.ai URL
    if url.startswith("https://r.jina.ai/"):
        jina_url = url
    else:
        jina_url = f"https://r.jina.ai/{url}"

    try:
        response = requests.get(jina_url, allow_redirects=True)
        response.raise_for_status()
        text = response.text

        if len(text) > MAX_RESPONSE_LENGTH:
            return (
                f"{text[:MAX_RESPONSE_LENGTH]}\n\n"
                "[Content truncated due to size limitations]"
            )

        return f"[URL]: {url}\n\n" + text

    except Exception as e:
        return f"Error fetching URL {url}: {e}"


def search_text(text: str, max_results: int = 3):
    """
    Search the web for the provided text
    Limit the search to max_results
    """

    dds = ddgs.duckduckgo_search.DDGS()
    return_str = ""
    results = []

    try:
        results = dds.text(text, max_results=max_results, backend="lite")
    except Exception as e:
        return_str += f"duck duck go search api error: [{e}]\n"

    for i, result in enumerate(results, start=1):
        return_str += f"{i}. {result['title']}\n   {result['href']}\n"

    return return_str


def search_text_alternative(text: str, max_results: int = 3):
    """
    Alternative web search for when the primary one fails.
    """


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
        return "DuckDuckGo search error:\n" + format_subproc_result(result)

    soup = BeautifulSoup(result.stdout, "html.parser")

    return soup.text


def search_images(text: str, max_results: int = 3):
    """
    Search the web for images that match the provided text
    Limit the search to max_results
    """


    dds = ddgs.duckduckgo_search.DDGS()
    results = dds.images(text, max_results=max_results)

    text = ""

    for i, result in enumerate(results or [], start=1):
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


    url = "https://api.openai.com/v1/images/generations"
    data = {
        "n": number,
        "model": model,
        "size": size,
        "prompt": prompt
    }

    try:
        response = requests.post(
            url,
            json=data,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
        )
        response.raise_for_status()
        return response.text

    except Exception as e:
        return f"Error creating images: {e}"


def read_file(path: str):
    """
    Read a file from the file system

    Args:
        path - relative path, ex. ./file.ext, or absolute path.
            Must contain folder, even if just ./
    """

    path = sanitize_path(path)

    try:
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return f"file not found: {path}"
    except IOError as e:
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

    try:
        # Ensure the directory exists, create if not
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception as e:
        return f"Error creating directories: {e}"

    # Capture previous contents (if any) so we can show a diff in the display
    prev_content = ""
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                prev_content = f.read()
    except Exception:
        prev_content = ""

    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(contents)
    except Exception as e:
        return f"Error writing to file: {e}"

    # If a live display is active, add a rich diff panel for this write
    try:
        display = get_tool_status_display()
        # compute unified diff
        import difflib
        old_lines = prev_content.splitlines(keepends=True)
        new_lines = contents.splitlines(keepends=True)
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"{os.path.basename(path)} (before)",
            tofile=f"{os.path.basename(path)} (after)",
            lineterm=''
        )
        diff_output = ''.join(diff)
        if diff_output:
            display.add_diff(os.path.basename(path), diff_output)
    except Exception:
        # Don't let display code break the tool
        pass

    return "Success"


def search_replace(path: str, old_text: str, new_text: str):
    """
    Search for exact text in a file and replace it with new text.

    Args:
        path: path to the target file
        old_text: the exact text to search for (must match exactly including whitespace)
        new_text: the text to replace it with

    Returns:
        A string indicating success or error message
    """

    file_path = sanitize_path(path)

    # Read the file contents
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return f"file not found: {file_path}"
    except IOError as e:
        return f"IOError while reading file: {e}"

    # Check if old_text exists in the file
    if old_text not in content:
        return f"Text not found in file: {file_path}"

    # Count occurrences
    occurrences = content.count(old_text)

    # Replace the text
    new_content = content.replace(old_text, new_text)

    # Generate a unified diff to show what changed
    import difflib
    old_lines = content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"{os.path.basename(file_path)} (before)",
        tofile=f"{os.path.basename(file_path)} (after)",
        lineterm='',
        n=3
    )

    diff_output = ''.join(diff)

    # Write back to file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    except IOError as e:
        return f"Error writing to file: {e}"

    # Also push a rich diff into the live display if available
    try:
        display = get_tool_status_display()
        if diff_output:
            display.add_diff(os.path.basename(file_path), diff_output)
    except Exception:
        pass

    return f"Successfully replaced {occurrences} occurrence(s)\n\nDiff:\n{diff_output}"

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


        try:
            if new_content == "":
                del messages[index]
            else:
                messages[index].content = new_content

            return f"message[{index}] successfully modified"
        except Exception as e:
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

    # Sanitize summary input to avoid injection (simple quote escape)
    safe_summary = summary.replace('"', '\"')

    # Use subprocess to call say asynchronously
    try:
        subprocess.Popen(['say', safe_summary])
        return "Speech synthesis started"
    except Exception as e:
        return f"Error starting speech synthesis: {e}"


def communicate(message: str):
    """
    Communicate intermediate thoughts, reasoning, or progress updates to the user.

    Use this tool to share your thought process as you work through complex tasks,
    explain what you're doing and why, or provide status updates during long-running operations.

    Args:
        message: The message to communicate to the user

    Returns:
        A confirmation that the message was delivered
    """
    # The actual display logic is handled in agent_runner.py
    # This tool just needs to return the message so it can be extracted
    return message


def spawn(task: str):
    """
    Spawn a new single-invocation AI agent to complete a specific task.

    Creates a fresh agent using aisuite that can work independently and
    return focused results. The spawned agent has access to basic tools.

    If environment variables OLLAMA_URL and OLLAMA_MODEL are set, the
    spawn will attempt to use the local Ollama HTTP API instead of
    aisuite/OpenAI. Otherwise it falls back to the existing aisuite
    client behavior.

    Args:
        task: The specific task description for the spawned agent to complete

    Returns:
        The output from the spawned agent's completion of the task
    """

    try:
        # Create a focused system prompt for the spawned agent
        spawn_system = (
            "You are a single-purpose AI agent spawned to complete one specific task. "
            "Focus solely on the given task, be concise, and provide actionable results. "
            "Complete the task efficiently and return your findings."
        )

        # If user configured Ollama, prefer that local model
        ollama_url = os.getenv("OLLAMA_URL")  # e.g., http://localhost:11434
        ollama_model = os.getenv("OLLAMA_MODEL")  # e.g., deepseek-r1:32b

        if ollama_url and ollama_model:
            try:
                # Prepare messages for Ollama chat completion
                messages = [
                    {"role": "system", "content": spawn_system},
                    {"role": "user", "content": task}
                ]

                # Ollama chat completions endpoint (best-effort) - support common endpoints
                endpoint = ollama_url.rstrip("/") + "/v1/chat/completions"

                payload = {
                    "model": ollama_model,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 2048
                }

                resp = requests.post(endpoint, json=payload, timeout=30)
                resp.raise_for_status()

                # Try to extract content from common response shapes
                data = resp.json()
                if isinstance(data, dict):
                    # Ollama often uses choices[0].message.content or choices[0].content
                    choices = data.get("choices") or []
                    if choices:
                        first = choices[0]
                        if isinstance(first, dict):
                            msg = first.get("message") or first.get("content") or {}
                            if isinstance(msg, dict):
                                content = msg.get("content") or msg.get("text")
                            else:
                                content = msg
                            if content:
                                return f"[SPAWNED AGENT OUTPUT]\n{content}"
                    # Fallback: try top-level text
                    if "text" in data:
                        return f"[SPAWNED AGENT OUTPUT]\n{data['text']}"

                return f"[SPAWNED AGENT OUTPUT]\n{resp.text}"

            except Exception as e:
                # If Ollama call fails, fall back to aisuite below
                print(f"OLLAMA spawn failed: {e}")

        # Default behavior: use aisuite client
        import aisuite as ai

        client = ai.Client()

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

        return f"[SPAWNED AGENT OUTPUT]\n{result}"

    except Exception as e:
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
