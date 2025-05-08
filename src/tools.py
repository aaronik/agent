import os
import time
import requests
import subprocess
import duckduckgo_search as ddgs
from src.util import extract_text, sanitize_path

MAX_RESPONSE_LENGTH = 1000000

# Fetching the OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def fetch(url: str):
    """Fetch content from the provided URL."""

    print(f"\n🔧 [fetch], url: [{url}]")

    response = requests.get(url)
    try:
        response.raise_for_status()
        text = extract_text(response.text)

        if len(text) > MAX_RESPONSE_LENGTH:
            print("⚠️ truncating response")
            return (
                f"{text[:MAX_RESPONSE_LENGTH]}\n\n"
                "[Content truncated due to size limitations]"
            )

        return (
            f"[URL]: {url}\n\n",
            text
        )

    except Exception as e:
        print(f"⚠️ status: [{response.status_code}], e: [{e}]")
        return f"Error fetching URL {url}: {e}"


def search_text(text: str, max_results: int = 3):
    """
    Search the web for the provided text
    Limit the search to max_results
    """

    print(f"\n🔧 [search], string: [{text}], max_results: [{max_results}]")

    dds = ddgs.duckduckgo_search.DDGS()

    results = []
    try:
        results = dds.text(text, max_results=max_results)
    except Exception as e:
        print(f"⚠️ duck duck go search error: [{e}] (retrying...)")
        time.sleep(0.5)

    text = ""

    for i, result in enumerate(results, start=1):
        print(f"🔗 {result['href']}")
        text += f"{i}. {result['title']}\n   {result['href']}\n"

    return text


def search_images(text: str, max_results: int = 3):
    """
    Search the web for images that match the provided text
    Limit the search to max_results
    """

    print(
        f"\n🔧 [search_images], string: [{text}], max_results: [{max_results}]"
    )

    dds = ddgs.duckduckgo_search.DDGS()
    results = dds.images(text, max_results=max_results)

    text = ""

    for i, result in enumerate(results or [], start=1):
        print(f"🖼️ {result['image']}")
        text += f"{i}. {result['title']}\n   {result['image']}\n"

    return text


def run_shell_command(cmd: str):
    """
    Run a shell command on the user's machine
    Use as many of these as needed to satisfy the user request
    """

    print(
        f"\n🔧 [run_shell_command], cmd: [{cmd}]"
    )

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    print(f"⮑  {result.returncode}")

    text = (
        "[STDOUT]",
        result.stdout,
        "[STDERR]",
        result.stderr,
        "[CODE]",
        result.returncode
    )

    return text


def printz(cmd: str):
    """
    Place a command on the user's command buffer.
    Use this when the user has asked for a command to do such and such.
    Named after the zsh print -z command.
    """

    print(
        f"\n🔧 [printz], cmd: [{cmd}]"
    )


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

    print(
        f"Generating {number} image(s) using [{model}], size: [{size}]",
        f"with prompt: {prompt}\n"
    )

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
        print(f"⚠️ status: [{response.status_code}]")
        return f"Error creating images: {e}"


def read_file(path: str):
    """
    Read a file from the file system

    Args:
        path - relative path, ex. ./file.ext, or absolute path.
            Must contain folder, even if just ./
    """

    path = sanitize_path(path)

    print(f"\n🔧 [read_file], path [{path}]")

    try:
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        return f"file not found: {path}"
    except IOError as e:
        print(f"❌ IOError while reading file: {e}")
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

    print(f"\n🔧 [write_file], path [{path}], contents: [{contents[:10]}]")

    try:
        # Ensure the directory exists, create if not
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception as e:
        print(f"❌ Error creating directories: {e}")
        return f"Error creating directories: {e}"

    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(contents)
    except Exception as e:
        print(f"❌ Error writing to file: {e}")
        return f"Error writing to file: {e}"

    return "Success"
