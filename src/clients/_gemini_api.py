# encoding = utf-8
import google.generativeai as genai
import time
import os

class GeminiVideoClient:
    def __init__(self, api_key):
        """Initializes the Gemini client with an API key."""
        genai.configure(api_key=api_key)

    def close(self):
        """A no-op method for API compatibility with session-based clients."""
        pass

    def upload_file(self, file_path, mime_type=None, display_name=None):
        """
        Uploads a file to Gemini and waits for it to be active.

        Args:
            file_path (str): The path to the local file.
            mime_type (str, optional): The MIME type of the file. Defaults to None.
            display_name (str, optional): The display name for the file. Defaults to None.
        
        Returns:
            A File object from the Gemini API.
        """
        if not display_name:
            display_name = os.path.basename(file_path)
            
        print(f"Uploading file: {file_path} as {display_name}")
        video_file = genai.upload_file(path=file_path, display_name=display_name, mime_type=mime_type)
        print(f"Completed upload: {video_file.name}. Waiting for it to be processed...")

        while video_file.state.name == "PROCESSING":
            time.sleep(5)
            video_file = genai.get_file(video_file.name)
        
        if video_file.state.name == "FAILED":
            raise Exception(f"File processing failed for {video_file.name}.")
        
        print(f"File {video_file.name} is now active.")
        return video_file

    def call_model(self, model_name, contents, generation_config, system_instruction=None, tools=None):
        """
        Calls the Gemini model with the provided contents and configuration.
        """
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            system_instruction=system_instruction,
            tools=tools
        )
        response = model.generate_content(contents)
        return response

    def delete_file(self, video_file):
        """Deletes a file from Gemini."""
        if video_file:
            genai.delete_file(video_file.name)
            print(f"File {video_file.name} deleted.")
